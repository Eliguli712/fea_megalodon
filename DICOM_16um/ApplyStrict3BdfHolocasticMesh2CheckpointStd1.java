import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ApplyStrict3BdfHolocasticMesh2CheckpointStd1 {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  // Prefer raw, high-resolution volumetric sources (no compressed gmsh variants).
  private static final String[] RAW_BDF_CANDIDATES =
      new String[] {
        "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf"
      };

  private static final String[] STUDIES =
      new String[] {"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};
  private static final String[] RUN_STUDIES = new String[] {"std1"};

  private static String ts() {
    return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
  }

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) {
      return false;
    }
    for (String t : tags) {
      if (needle.equals(t)) {
        return true;
      }
    }
    return false;
  }

  private static String safeMsg(Throwable t) {
    if (t == null) {
      return "";
    }
    String m = t.getMessage();
    if (m == null || m.isEmpty()) {
      return t.getClass().getSimpleName();
    }
    return m.replace('\n', ' ').replace('\r', ' ');
  }

  private static int countAllEntities(Model model, int dim, String tag) {
    try {
      try {
        model.component("comp1").selection().remove(tag);
      } catch (Exception ignored) {
      }
      model.component("comp1").selection().create(tag, "Explicit");
      model.component("comp1").selection(tag).geom("geom1", dim);
      model.component("comp1").selection(tag).all();
      int[] e = model.component("comp1").selection(tag).entities();
      return e == null ? 0 : e.length;
    } catch (Exception ignored) {
      return -1;
    }
  }

  private static double evalMaxVolume(Model model, String dset, String expr, String tag) {
    try {
      try {
        model.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      model.result().numerical().create(tag, "MaxVolume");
      model.result().numerical(tag).set("data", dset);
      model.result().numerical(tag).set("expr", new String[] {expr});
      model.result().numerical(tag).selection().all();
      model.result().numerical(tag).setResult();
      double[][] r = model.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        return r[0][0];
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static void configureMesh1FromPart2(Model model, List<String> logs) {
    if (!hasTag(model.component("comp1").mesh().tags(), "mesh1")) {
      throw new RuntimeException("comp1/mesh1 is missing.");
    }
    if (!hasTag(model.component("comp1").mesh("mesh1").feature().tags(), "impmsh")) {
      model.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    }
    model.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart2");
    try {
      model.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      model.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    try {
      model.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
    } catch (Exception ignored) {
    }
    model.component("comp1").mesh("mesh1").run("impmsh");
    logs.add("MESH_RUN|comp1/mesh1/impmsh|ok=true");
  }

  private static void clearMesh2Features(Model model) {
    String[] ftags = model.component("comp1").mesh("mesh2").feature().tags();
    if (ftags == null) {
      return;
    }
    for (String t : ftags) {
      try {
        model.component("comp1").mesh("mesh2").feature().remove(t);
      } catch (Exception ignored) {
      }
    }
  }

  private static int runMeshPart2(Model model, List<String> logs) {
    if (!hasTag(model.component("comp1").mesh().tags(), "mesh2")) {
      throw new RuntimeException("comp1/mesh2 is missing.");
    }

    // Preferred path: keep raw high-resolution domain elements by importing mesh2 from mpart2.
    try {
      clearMesh2Features(model);
      model.component("comp1").mesh("mesh2").feature().create("impmsh", "Import");
      model.component("comp1").mesh("mesh2").feature("impmsh").set("source", "sequence");
      model.component("comp1").mesh("mesh2").feature("impmsh").set("sequence", "mpart2");
      try {
        model.component("comp1").mesh("mesh2").feature("impmsh").set("buildsource", "on");
      } catch (Exception ignored) {
      }
      try {
        model.component("comp1").mesh("mesh2").feature("impmsh").set("domelemsequence", "on");
      } catch (Exception ignored) {
      }
      model.component("comp1").mesh("mesh2").run("impmsh");
      logs.add("MESH_RUN|comp1/mesh2|ok=true|mode=import_from_mpart2");

      int tetImp = -1;
      try {
        tetImp = model.component("comp1").mesh("mesh2").getNumElem("tet");
      } catch (Exception ignored) {
      }
      logs.add("MESH2_TET_COUNT|" + tetImp + "|mode=import_from_mpart2");
      if (tetImp > 0) {
        return tetImp;
      }
    } catch (Exception e) {
      logs.add("MESH_RUN|comp1/mesh2|ok=false|mode=import_from_mpart2|err=" + safeMsg(e));
    }

    // Fallback path: geometric FreeTet remeshing on mesh2.
    try {
      clearMesh2Features(model);
      model.component("comp1").mesh("mesh2").feature().create("size1", "Size");
      model.component("comp1").mesh("mesh2").feature().create("ftet1", "FreeTet");
      model.component("comp1").mesh("mesh2").feature("size1").selection().geom("geom1", 3);
      model.component("comp1").mesh("mesh2").feature("size1").selection().all();
    } catch (Exception ignored) {
    }
    try {
      model.component("comp1").mesh("mesh2").feature("ftet1").selection().geom("geom1", 3);
      model.component("comp1").mesh("mesh2").feature("ftet1").selection().all();
    } catch (Exception ignored) {
    }

    model.component("comp1").mesh("mesh2").run();
    logs.add("MESH_RUN|comp1/mesh2|ok=true|mode=freetet");

    int tet = -1;
    try {
      tet = model.component("comp1").mesh("mesh2").getNumElem("tet");
    } catch (Exception ignored) {
    }
    logs.add("MESH2_TET_COUNT|" + tet + "|mode=freetet");
    return tet;
  }

  private static int configureRawUncompressedImport(Model model, List<String> logs) {
    if (!hasTag(model.mesh().tags(), "mpart2")) {
      throw new RuntimeException("Global mesh part mpart2 is missing.");
    }
    if (!hasTag(model.mesh("mpart2").feature().tags(), "imp1")) {
      model.mesh("mpart2").feature().create("imp1", "Import");
    }
    for (String bdf : RAW_BDF_CANDIDATES) {
      try {
        model.mesh("mpart2").feature("imp1").set("source", "nastran");
        model.mesh("mpart2").feature("imp1").set("filename", bdf);
        model.mesh("mpart2").feature("imp1").set("createdom", "on");
        model.mesh("mpart2").feature("imp1").set("facepartition", "minimal");
        // Run only the import feature to avoid destructive remeshing passes.
        model.mesh("mpart2").run("imp1");
        logs.add("MESH_RUN|mpart2/imp1|ok=true|bdf=" + bdf);
      } catch (Exception e) {
        logs.add("MESH_RUN|mpart2/imp1|ok=false|bdf=" + bdf + "|err=" + safeMsg(e));
        continue;
      }

      try {
        configureMesh1FromPart2(model, logs);
      } catch (Exception e) {
        logs.add("MESH_RUN|comp1/mesh1/impmsh|ok=false|bdf=" + bdf + "|err=" + safeMsg(e));
        continue;
      }

      try {
        int tet = runMeshPart2(model, logs);
        if (tet > 0) {
          logs.add("RAW_BDF_IN_USE|" + bdf);
          return tet;
        }
      } catch (Exception e) {
        logs.add("MESH_RUN|comp1/mesh2|ok=false|bdf=" + bdf + "|err=" + safeMsg(e));
      }
    }
    throw new RuntimeException("All raw high-resolution BDF candidates failed to generate nonzero mesh2.");
  }

  private static void bindStudiesToMesh2(Model model, List<String> logs) {
    for (String std : STUDIES) {
      if (!hasTag(model.study().tags(), std)) {
        logs.add("STUDY_SKIP|" + std + "|reason=missing");
        continue;
      }
      try {
        model.study(std).feature("stat").set("mesh", new String[][] {{"geom1", "mesh2"}});
        logs.add("STUDY_BIND|" + std + "|mesh=geom1/mesh2|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_BIND|" + std + "|mesh=geom1/mesh2|ok=false|err=" + safeMsg(e));
      }
      try {
        model.study(std).feature("stat").set("plot", "off");
      } catch (Exception ignored) {
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    logs.add("MODEL|" + MPH);
    logs.add("RAW_DATA_COMPRESSION|off");
    logs.add("CHUNK_POLICY|checkpoint_config_then_single_study_with_gc");
    logs.add("RAW_BDF_CANDIDATES|" + String.join(",", RAW_BDF_CANDIDATES));

    int b0 = countAllEntities(model, 2, "tmp_bnd_before_uncompressed");
    int d0 = countAllEntities(model, 3, "tmp_dom_before_uncompressed");
    logs.add("ENTITY_COUNTS_BEFORE|bnd=" + b0 + "|dom=" + d0);

    int tet = configureRawUncompressedImport(model, logs);

    int b1 = countAllEntities(model, 2, "tmp_bnd_after_uncompressed");
    int d1 = countAllEntities(model, 3, "tmp_dom_after_uncompressed");
    logs.add("ENTITY_COUNTS_AFTER|bnd=" + b1 + "|dom=" + d1);

    bindStudiesToMesh2(model, logs);

    String checkpointBackup = MPH + ".bak-" + ts();
    model.save(checkpointBackup);
    logs.add("CHECKPOINT_BACKUP|" + checkpointBackup);
    model.save(MPH);
    logs.add("CHECKPOINT_SAVED|" + MPH);

    Map<String, String> dset = new LinkedHashMap<String, String>();
    dset.put("std1", "dset6");
    dset.put("std_nh", "dset1");
    dset.put("std_og", "dset2");
    dset.put("std_mr2", "dset3");
    dset.put("std_mr5", "dset4");
    dset.put("std_pr", "dset5");

    int good = 0;
    for (String std : RUN_STUDIES) {
      if (!hasTag(model.study().tags(), std)) {
        continue;
      }

      try {
        model.study(std).run();
        logs.add("STUDY_RUN|" + std + "|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_RUN|" + std + "|ok=false|err=" + safeMsg(e));
        continue;
      }

      String ds = dset.get(std);
      double vm = evalMaxVolume(model, ds, "solid.mises", "mxvm_mesh2_uncompressed_" + std);
      double um = evalMaxVolume(model, ds, "sqrt(u^2+v^2+w^2)", "mxu_mesh2_uncompressed_" + std);
      boolean ok =
          Double.isFinite(vm)
              && Double.isFinite(um)
              && Math.abs(vm) > 1e-12
              && Math.abs(um) > 1e-15
              && tet > 0;
      if (ok) {
        good++;
      }
      logs.add(
          "CHECK|study="
              + std
              + "|dataset="
              + ds
              + "|vm="
              + vm
              + "|um="
              + um
              + "|finite_nonzero="
              + ok);

      try {
        System.gc();
      } catch (Exception ignored) {
      }
    }

    String backup = MPH + ".bak-" + ts();
    model.save(backup);
    logs.add("BACKUP|" + backup);
    model.save(MPH);
    logs.add("SAVED|" + MPH);

    logs.add(
        "SUMMARY|finite_nonzero_studies="
            + good
            + "|total_target_studies="
            + RUN_STUDIES.length
            + "|bound_studies="
            + STUDIES.length);
    for (String line : logs) {
      System.out.println(line);
    }
  }
}
