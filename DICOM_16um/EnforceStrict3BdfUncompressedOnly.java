import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EnforceStrict3BdfUncompressedOnly {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";
  private static final String BDF_UNCOMPRESSED =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf";
  private static final String[] STUDIES =
      new String[] {"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};

  private static String ts() {
    return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
  }

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) {
      if (needle.equals(t)) return true;
    }
    return false;
  }

  private static String safeMsg(Throwable t) {
    if (t == null) return "";
    String m = t.getMessage();
    if (m == null || m.isEmpty()) return t.getClass().getSimpleName();
    return m.replace('\n', ' ').replace('\r', ' ');
  }

  private static void clearMeshFeatures(MeshFeatureList fl) {
    String[] tags;
    try {
      tags = fl.tags();
    } catch (Exception ignored) {
      return;
    }
    if (tags == null) return;
    for (String t : tags) {
      if ("fin".equals(t)) continue;
      try {
        fl.remove(t);
      } catch (Exception ignored) {
      }
    }
  }

  private static void configureCompMeshFromMpart2(Model m, String compTag, String meshTag, List<String> logs) {
    if (!hasTag(m.component(compTag).mesh().tags(), meshTag)) {
      logs.add("MESH_SKIP|" + compTag + "/" + meshTag + "|reason=missing");
      return;
    }
    clearMeshFeatures(m.component(compTag).mesh(meshTag).feature());
    if (!hasTag(m.component(compTag).mesh(meshTag).feature().tags(), "impmsh")) {
      m.component(compTag).mesh(meshTag).feature().create("impmsh", "Import");
    }
    m.component(compTag).mesh(meshTag).feature("impmsh").set("source", "sequence");
    m.component(compTag).mesh(meshTag).feature("impmsh").set("sequence", "mpart2");
    try {
      m.component(compTag).mesh(meshTag).feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      m.component(compTag).mesh(meshTag).feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    m.component(compTag).mesh(meshTag).run("impmsh");
    logs.add("MESH_RUN|" + compTag + "/" + meshTag + "/impmsh|ok=true|source=mpart2");

    try {
      int tet = m.component(compTag).mesh(meshTag).getNumElem("tet");
      logs.add("MESH_TET|" + compTag + "/" + meshTag + "|" + tet);
    } catch (Exception e) {
      logs.add("MESH_TET|" + compTag + "/" + meshTag + "|ERR|" + safeMsg(e));
    }
  }

  private static void ensureMpart2Uncompressed(Model m, List<String> logs) {
    if (!hasTag(m.mesh().tags(), "mpart2")) {
      if (!hasTag(m.mesh().tags(), "mpart1")) {
        throw new RuntimeException("Cannot create mpart2: mpart1 missing.");
      }
      m.mesh().duplicate("mpart2", "mpart1");
      logs.add("MESH_CREATE|mpart2|mode=duplicate_mpart1|ok=true");
    }

    if (!hasTag(m.mesh("mpart2").feature().tags(), "imp1")) {
      m.mesh("mpart2").feature().create("imp1", "Import");
    }

    m.mesh("mpart2").feature("imp1").set("source", "nastran");
    m.mesh("mpart2").feature("imp1").set("filename", BDF_UNCOMPRESSED);
    m.mesh("mpart2").feature("imp1").set("createdom", "on");
    try {
      m.mesh("mpart2").feature("imp1").set("facepartition", "minimal");
    } catch (Exception ignored) {
    }
    m.mesh("mpart2").run("imp1");
    logs.add("MESH_RUN|mpart2/imp1|ok=true|bdf=" + BDF_UNCOMPRESSED);
  }

  private static void ensureComp2(Model m, List<String> logs) {
    if (!hasTag(m.component().tags(), "comp2")) {
      m.component().create("comp2");
      logs.add("COMP2_CREATE|ok=true");
    }
    try {
      m.component("comp2").label("Component 2");
    } catch (Exception ignored) {
    }
    if (!hasTag(m.component("comp2").geom().tags(), "geom2")) {
      m.component("comp2").geom().create("geom2", 3);
      logs.add("COMP2_GEOM_CREATE|geom2|ok=true");
    }
    if (!hasTag(m.component("comp2").mesh().tags(), "mesh3")) {
      m.component("comp2").mesh().create("mesh3", "geom2");
      logs.add("COMP2_MESH_CREATE|mesh3|ok=true");
    }

    configureCompMeshFromMpart2(m, "comp2", "mesh3", logs);

    try {
      m.component("comp2").geometricModel("mesh/mesh3");
      logs.add("COMP2_GEOMETRIC_MODEL|mesh/mesh3");
    } catch (Exception e) {
      logs.add("COMP2_GEOMETRIC_MODEL|ERR|" + safeMsg(e));
    }
  }

  private static void bindStudiesComp2First(Model m, List<String> logs) {
    for (String st : STUDIES) {
      if (!hasTag(m.study().tags(), st)) {
        logs.add("STUDY_SKIP|" + st + "|reason=missing");
        continue;
      }
      try {
        try {
          m.study(st).feature("stat").set("mesh", new String[][] {{"geom2", "mesh3"}, {"geom1", "mesh2"}});
        } catch (Exception ignored) {
          m.study(st).feature("stat").set("mesh", new String[] {"geom2", "mesh3", "geom1", "mesh2"});
        }
        try {
          m.study(st).feature("stat").set("plot", "off");
        } catch (Exception ignored) {
        }
        logs.add("STUDY_BIND|" + st + "|mesh=geom2/mesh3,geom1/mesh2|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_BIND|" + st + "|ok=false|err=" + safeMsg(e));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    logs.add("MODEL|" + MPH);
    logs.add("STRICT_BDF|" + BDF_UNCOMPRESSED);
    logs.add("COMPONENTS_BEFORE|" + Arrays.toString(m.component().tags()));

    ensureMpart2Uncompressed(m, logs);
    configureCompMeshFromMpart2(m, "comp1", "mesh1", logs);
    configureCompMeshFromMpart2(m, "comp1", "mesh2", logs);
    ensureComp2(m, logs);
    bindStudiesComp2First(m, logs);

    String backup = MPH + ".bak-" + ts();
    m.save(backup);
    logs.add("BACKUP|" + backup);
    m.save(MPH);
    logs.add("SAVED|" + MPH);

    logs.add("COMPONENTS_AFTER|" + Arrays.toString(m.component().tags()));
    for (String line : logs) {
      System.out.println(line);
    }
  }
}
