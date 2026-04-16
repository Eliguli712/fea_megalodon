import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LoadToothUncompressedTwoMeshesAndRetest {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";
  private static final String TOOTH_BDF =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf";
  private static final String[] STUDIES =
      new String[]{"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};

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

  private static String safeType(ResultFeature rf) {
    try {
      return rf.getType();
    } catch (Exception ignored) {
      return "";
    }
  }

  private static String safeString(PropFeature pf, String key) {
    try {
      String v = pf.getString(key);
      return v == null ? "" : v;
    } catch (Exception ignored) {
      return "";
    }
  }

  private static void ensureMeshImportFromTooth(Model model, List<String> logs) {
    String oldFile = "";
    try {
      oldFile = model.mesh("mpart1").feature("imp1").getString("filename");
    } catch (Exception ignored) {
    }

    model.mesh("mpart1").feature("imp1").set("source", "nastran");
    model.mesh("mpart1").feature("imp1").set("filename", TOOTH_BDF);
    model.mesh("mpart1").feature("imp1").set("createdom", "on");
    model.mesh("mpart1").feature("imp1").set("facepartition", "minimal");

    logs.add("MPART1_IMP1_FILE_OLD|" + oldFile);
    logs.add("MPART1_IMP1_FILE_NEW|" + TOOTH_BDF);

    model.mesh("mpart1").run("imp1");
    logs.add("MESH_RUN|mpart1/imp1|ok=true");
    model.mesh("mpart1").run();
    logs.add("MESH_RUN|mpart1|ok=true");
  }

  private static void runMesh1AndMesh2(Model model, List<String> logs) {
    if (hasTag(model.component("comp1").mesh().tags(), "mesh1")) {
      if (!hasTag(model.component("comp1").mesh("mesh1").feature().tags(), "impmsh")) {
        model.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
      }
      model.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
      model.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
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
      model.component("comp1").mesh("mesh1").run();
      logs.add("MESH_RUN|comp1/mesh1|ok=true");
    }

    if (hasTag(model.component("comp1").mesh().tags(), "mesh2")) {
      model.component("comp1").mesh("mesh2").run();
      logs.add("MESH_RUN|comp1/mesh2|ok=true");
    }

    if (hasTag(model.mesh().tags(), "mesh2")) {
      model.mesh("mesh2").run();
      logs.add("MESH_RUN|global/mesh2|ok=true");
    }
  }

  private static void bindStudiesToMesh2(Model model, List<String> logs) {
    for (String std : STUDIES) {
      if (!hasTag(model.study().tags(), std)) {
        logs.add("STUDY_SKIP|" + std + "|reason=missing");
        continue;
      }
      try {
        model.study(std).feature("stat").set("mesh", new String[][]{{"geom1", "mesh2"}});
        logs.add("STUDY_BIND|" + std + "|mesh=mesh2|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_BIND|" + std + "|mesh=mesh2|ok=false|err=" + safeMsg(e));
      }
      try {
        model.study(std).run();
        logs.add("STUDY_RUN|" + std + "|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_RUN|" + std + "|ok=false|err=" + safeMsg(e));
      }
    }
  }

  private static double evalMaxVolume(Model model, String dset, String expr) {
    final String tag = "vm_chk2";
    try {
      try {
        model.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      model.result().numerical().create(tag, "MaxVolume");
      model.result().numerical(tag).set("data", dset);
      model.result().numerical(tag).set("expr", new String[]{expr});
      model.result().numerical(tag).setResult();
      double[][] r = model.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        return r[0][0];
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static boolean hasMisesSurface(ResultFeature pg) {
    try {
      String[] kids = pg.feature().tags();
      if (kids == null) {
        return false;
      }
      for (String k : kids) {
        ResultFeature rf = pg.feature(k);
        if (!"Surface".equals(safeType(rf))) {
          continue;
        }
        String expr = safeString(rf, "expr");
        if (expr.contains("mises")) {
          return true;
        }
        try {
          String[] arr = rf.getStringArray("expr");
          if (arr != null) {
            for (String e : arr) {
              if (e != null && e.contains("mises")) {
                return true;
              }
            }
          }
        } catch (Exception ignored) {
        }
      }
    } catch (Exception ignored) {
    }
    return false;
  }

  private static void runMisesPlots(Model model, List<String> logs) {
    for (String pgTag : model.result().tags()) {
      ResultFeature pg;
      try {
        pg = model.result(pgTag);
      } catch (Exception e) {
        continue;
      }
      String type = safeType(pg);
      if (!type.startsWith("PlotGroup")) {
        continue;
      }
      if (!hasMisesSurface(pg)) {
        continue;
      }
      try {
        pg.run();
        logs.add("PLOT_RUN|" + pgTag + "|ok=true");
      } catch (Exception e) {
        logs.add("PLOT_RUN|" + pgTag + "|ok=false|err=" + safeMsg(e));
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
    logs.add("TOOTH_BDF|" + TOOTH_BDF);
    logs.add("GLOBAL_MESH_TAGS|" + Arrays.toString(model.mesh().tags()));
    logs.add("COMP1_MESH_TAGS|" + Arrays.toString(model.component("comp1").mesh().tags()));

    ensureMeshImportFromTooth(model, logs);
    runMesh1AndMesh2(model, logs);
    bindStudiesToMesh2(model, logs);

    String[] dsets = new String[]{"dset6", "dset1", "dset2", "dset3", "dset4", "dset5"};
    for (String d : dsets) {
      double vm = evalMaxVolume(model, d, "solid.mises");
      logs.add("EVAL_VM|maxvol|dataset=" + d + "|value=" + vm + "|finite=" + Double.isFinite(vm));
      if (Double.isFinite(vm)) {
        break;
      }
    }

    runMisesPlots(model, logs);

    String backup = MPH + ".bak-" + ts();
    model.save(backup);
    logs.add("BACKUP|" + backup);
    model.save(MPH);
    logs.add("SAVED|" + MPH);

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
