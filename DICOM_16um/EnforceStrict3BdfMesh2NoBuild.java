import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EnforceStrict3BdfMesh2NoBuild {
  private static final String MPH = "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";
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

  private static void configureCompMeshSequence(Model m, String comp, String mesh, String seq, List<String> logs) {
    if (!hasTag(m.component(comp).mesh().tags(), mesh)) {
      logs.add("MESH_SKIP|" + comp + "/" + mesh + "|reason=missing");
      return;
    }
    clearMeshFeatures(m.component(comp).mesh(mesh).feature());
    if (!hasTag(m.component(comp).mesh(mesh).feature().tags(), "impmsh")) {
      m.component(comp).mesh(mesh).feature().create("impmsh", "Import");
    }
    m.component(comp).mesh(mesh).feature("impmsh").set("source", "sequence");
    m.component(comp).mesh(mesh).feature("impmsh").set("sequence", seq);
    try {
      m.component(comp).mesh(mesh).feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      m.component(comp).mesh(mesh).feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    logs.add("MESH_CONFIG|" + comp + "/" + mesh + "/impmsh|source=" + seq + "|build=deferred");
  }

  private static void ensureComp2Manual(Model m, List<String> logs) {
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

    clearMeshFeatures(m.component("comp2").mesh("mesh3").feature());
    if (!hasTag(m.component("comp2").mesh("mesh3").feature().tags(), "impmsh")) {
      m.component("comp2").mesh("mesh3").feature().create("impmsh", "Import");
    }
    m.component("comp2").mesh("mesh3").feature("impmsh").set("source", "sequence");
    m.component("comp2").mesh("mesh3").feature("impmsh").set("sequence", "mpart2");
    try {
      m.component("comp2").mesh("mesh3").feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      m.component("comp2").mesh("mesh3").feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    logs.add("MESH_CONFIG|comp2/mesh3/impmsh|source=mpart2|build=deferred");

    try {
      m.component("comp2").geometricModel("mesh/mesh3");
      logs.add("COMP2_GEOMETRIC_MODEL|mesh/mesh3");
    } catch (Exception e) {
      logs.add("COMP2_GEOMETRIC_MODEL|ERR|" + safeMsg(e));
    }
  }

  private static void bindStudies(Model m, List<String> logs) {
    for (String st : STUDIES) {
      if (!hasTag(m.study().tags(), st)) {
        logs.add("STUDY_SKIP|" + st + "|reason=missing");
        continue;
      }
      try {
        try {
          m.study(st).feature("stat").set("mesh", new String[][] {{"geom1", "mesh2"}, {"geom2", "mesh3"}});
        } catch (Exception ignored) {
          m.study(st).feature("stat").set("mesh", new String[] {"geom1", "mesh2", "geom2", "mesh3"});
        }
        try {
          m.study(st).feature("stat").set("plot", "off");
        } catch (Exception ignored) {
        }
        logs.add("STUDY_BIND|" + st + "|mesh=geom1/mesh2,geom2/mesh3|ok=true");
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
    logs.add("COMPONENTS_BEFORE|" + Arrays.toString(m.component().tags()));

    if (!hasTag(m.mesh().tags(), "mpart2")) {
      throw new RuntimeException("Global mesh part mpart2 is missing.");
    }

    configureCompMeshSequence(m, "comp1", "mesh1", "mpart2", logs);
    configureCompMeshSequence(m, "comp1", "mesh2", "mpart2", logs);
    ensureComp2Manual(m, logs);
    bindStudies(m, logs);

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
