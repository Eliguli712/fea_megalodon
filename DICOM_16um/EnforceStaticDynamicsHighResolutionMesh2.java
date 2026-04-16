import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EnforceStaticDynamicsHighResolutionMesh2 {
  private static final String MPH = "DICOM_16um/exports/static_dynamics_high_resolution.mph";

  private static final String[] BDF_CANDIDATES =
      new String[] {
        "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf",
        "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_raw_compare_comsol_tet_vol.bdf",
        "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_smoothed_compare_comsol_tet_vol.bdf",
        "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_comsol_tet_vol.bdf"
      };

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

  private static void ensureMpart2Import(Model m, List<String> logs) {
    if (!hasTag(m.mesh().tags(), "mpart2")) {
      if (!hasTag(m.mesh().tags(), "mpart1")) {
        throw new RuntimeException("Cannot create mpart2: mpart1 is missing.");
      }
      try {
        m.mesh().duplicate("mpart2", "mpart1");
        logs.add("CREATE|global/mpart2|ok=true|mode=duplicate_mpart1");
      } catch (Exception e) {
        throw new RuntimeException("Cannot create mpart2 by duplicating mpart1: " + safeMsg(e));
      }
    }
    if (!hasTag(m.mesh("mpart2").feature().tags(), "imp1")) {
      m.mesh("mpart2").feature().create("imp1", "Import");
    }

    for (String bdf : BDF_CANDIDATES) {
      try {
        m.mesh("mpart2").feature("imp1").set("source", "nastran");
        m.mesh("mpart2").feature("imp1").set("filename", bdf);
        m.mesh("mpart2").feature("imp1").set("createdom", "on");
        try {
          m.mesh("mpart2").feature("imp1").set("facepartition", "minimal");
        } catch (Exception ignored) {
        }
        logs.add("MESH_CONFIG|global/mpart2/imp1|ok=true|bdf=" + bdf + "|build=deferred");
        return;
      } catch (Exception e) {
        logs.add("MESH_CONFIG|global/mpart2/imp1|ok=false|bdf=" + bdf + "|err=" + safeMsg(e));
      }
    }
    throw new RuntimeException("Failed to configure any BDF candidate for mpart2.");
  }

  private static void bindGlobalMeshToPart(Model m, String meshTag, List<String> logs) {
    if (!hasTag(m.mesh().tags(), meshTag)) {
      logs.add("MESH_SKIP|global/" + meshTag + "|reason=missing");
      return;
    }
    clearMeshFeatures(m.mesh(meshTag).feature());
    if (!hasTag(m.mesh(meshTag).feature().tags(), "impmsh")) {
      m.mesh(meshTag).feature().create("impmsh", "Import");
    }
    m.mesh(meshTag).feature("impmsh").set("source", "sequence");
    m.mesh(meshTag).feature("impmsh").set("sequence", "mpart2");
    try {
      m.mesh(meshTag).feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      m.mesh(meshTag).feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    logs.add("MESH_CONFIG|global/" + meshTag + "/impmsh|ok=true|source=mpart2|build=deferred");
  }

  private static void bindComp1MeshToPart(Model m, String meshTag, List<String> logs) {
    if (!hasTag(m.component("comp1").mesh().tags(), meshTag)) {
      logs.add("MESH_SKIP|comp1/" + meshTag + "|reason=missing");
      return;
    }
    clearMeshFeatures(m.component("comp1").mesh(meshTag).feature());
    if (!hasTag(m.component("comp1").mesh(meshTag).feature().tags(), "impmsh")) {
      m.component("comp1").mesh(meshTag).feature().create("impmsh", "Import");
    }
    m.component("comp1").mesh(meshTag).feature("impmsh").set("source", "sequence");
    m.component("comp1").mesh(meshTag).feature("impmsh").set("sequence", "mpart2");
    try {
      m.component("comp1").mesh(meshTag).feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").mesh(meshTag).feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    logs.add("MESH_CONFIG|comp1/" + meshTag + "/impmsh|ok=true|source=mpart2|build=deferred");
  }

  private static void bindAllStudiesToMesh2(Model m, List<String> logs) {
    String[] studies;
    try {
      studies = m.study().tags();
    } catch (Exception e) {
      logs.add("STUDY_ENUM|ok=false|err=" + safeMsg(e));
      return;
    }
    logs.add("STUDIES|" + Arrays.toString(studies));
    for (String st : studies) {
      try {
        m.study(st).feature("stat").set("mesh", new String[][] {{"geom1", "mesh2"}});
        try {
          m.study(st).feature("stat").set("plot", "off");
        } catch (Exception ignored) {
        }
        logs.add("STUDY_BIND|" + st + "|mesh=geom1/mesh2|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_BIND|" + st + "|mesh=geom1/mesh2|ok=false|err=" + safeMsg(e));
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
    logs.add("GLOBAL_MESH_BEFORE|" + Arrays.toString(m.mesh().tags()));
    logs.add("COMP1_MESH_BEFORE|" + Arrays.toString(m.component("comp1").mesh().tags()));

    ensureMpart2Import(m, logs);
    // Study solve path uses component mesh bindings; global mesh rewrites are skipped
    // to avoid unnecessary remeshing overhead.
    bindComp1MeshToPart(m, "mesh1", logs);
    bindComp1MeshToPart(m, "mesh2", logs);
    bindAllStudiesToMesh2(m, logs);

    String backup = MPH + ".bak-" + ts();
    m.save(backup);
    logs.add("BACKUP|" + backup);
    m.save(MPH);
    logs.add("SAVED|" + MPH);

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
