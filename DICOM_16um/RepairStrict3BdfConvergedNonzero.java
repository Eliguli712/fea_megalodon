import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class RepairStrict3BdfConvergedNonzero {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";
  private static final String BDF_VOL =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_comsol_tet_vol.bdf";

  private static final String SEL_DOM = "sel_dom_all_auto";
  private static final String SEL_FIX = "sel_tail_fix_auto";
  private static final String SEL_LOAD = "sel_front_load_auto";

  private static final String FIX_TAG = "fix_auto";
  private static final String LOAD_TAG = "bndl_auto";
  private static final String BODY_TAG = "body_auto";
  private static final String RMS_TAG = "rms_auto";

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

  private static void safeActivate(Model m, String tag, boolean active) {
    try {
      m.component("comp1").physics("solid").feature(tag).active(active);
    } catch (Exception ignored) {
    }
  }

  private static int countSelectionEntities(Model m, String tag) {
    try {
      int[] e = m.component("comp1").selection(tag).entities();
      return e == null ? 0 : e.length;
    } catch (Exception ignored) {
      return -1;
    }
  }

  private static int countAll(Model m, int dim, String tag) {
    try {
      try {
        m.component("comp1").selection().remove(tag);
      } catch (Exception ignored) {
      }
      m.component("comp1").selection().create(tag, "Explicit");
      m.component("comp1").selection(tag).geom("geom1", dim);
      m.component("comp1").selection(tag).all();
      int[] e = m.component("comp1").selection(tag).entities();
      return e == null ? 0 : e.length;
    } catch (Exception ignored) {
      return -1;
    }
  }

  private static void ensureSelections(Model m, List<String> logs) {
    try {
      m.component("comp1").selection().remove(SEL_FIX);
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").selection().remove(SEL_LOAD);
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").selection().remove(SEL_DOM);
    } catch (Exception ignored) {
    }

    m.component("comp1").selection().create(SEL_DOM, "Explicit");
    m.component("comp1").selection(SEL_DOM).geom("geom1", 3);
    m.component("comp1").selection(SEL_DOM).all();

    // Tail support (lowest z band).
    m.component("comp1").selection().create(SEL_FIX, "Box");
    m.component("comp1").selection(SEL_FIX).set("entitydim", 2);
    m.component("comp1").selection(SEL_FIX).set("xmin", -1e9);
    m.component("comp1").selection(SEL_FIX).set("xmax", 1e9);
    m.component("comp1").selection(SEL_FIX).set("ymin", -1e9);
    m.component("comp1").selection(SEL_FIX).set("ymax", 1e9);
    m.component("comp1").selection(SEL_FIX).set("zmin", -1e9);
    m.component("comp1").selection(SEL_FIX).set("zmax", 5.0);

    // Front load (highest z band).
    m.component("comp1").selection().create(SEL_LOAD, "Box");
    m.component("comp1").selection(SEL_LOAD).set("entitydim", 2);
    m.component("comp1").selection(SEL_LOAD).set("xmin", -1e9);
    m.component("comp1").selection(SEL_LOAD).set("xmax", 1e9);
    m.component("comp1").selection(SEL_LOAD).set("ymin", -1e9);
    m.component("comp1").selection(SEL_LOAD).set("ymax", 1e9);
    m.component("comp1").selection(SEL_LOAD).set("zmin", 40.0);
    m.component("comp1").selection(SEL_LOAD).set("zmax", 1e9);

    logs.add("SEL_COUNT|" + SEL_DOM + "|" + countSelectionEntities(m, SEL_DOM));
    logs.add("SEL_COUNT|" + SEL_FIX + "|" + countSelectionEntities(m, SEL_FIX));
    logs.add("SEL_COUNT|" + SEL_LOAD + "|" + countSelectionEntities(m, SEL_LOAD));
  }

  private static void ensurePhysics(Model m, List<String> logs) {
    m.param().set("force_density_z", "2.5e5[N/m^3]");
    m.param().set("p_front", "8.0e5[Pa]");

    m.param().set("stvk_E", "1.5e8[Pa]");
    m.param().set("stvk_nu", "0.30");
    m.param().set("kappa_bulk", "2.5e8[Pa]");
    m.param().set("mr2_c10", "1.6e7[Pa]");
    m.param().set("mr2_c01", "4.0e6[Pa]");
    m.param().set("mr5_c10", "1.2e7[Pa]");
    m.param().set("mr5_c01", "3.0e6[Pa]");
    m.param().set("mr5_c20", "2.0e6[Pa]");
    m.param().set("mr5_c11", "1.5e6[Pa]");
    m.param().set("mr5_c02", "8.0e5[Pa]");

    // Ensure constitutive models target the domain selection, not empty selections.
    String[] matTags = new String[]{"lemm1", "hmm_nh", "hmm_og", "hmm_mr2", "hmm_mr5"};
    for (String t : matTags) {
      try {
        m.component("comp1").physics("solid").feature(t).selection().named(SEL_DOM);
        logs.add("PHYS_SEL|" + t + "|" + SEL_DOM);
      } catch (Exception e) {
        logs.add("PHYS_SEL_ERR|" + t + "|" + safeMsg(e));
      }
    }

    try {
      m.component("comp1").physics("solid").feature().remove(FIX_TAG);
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").physics("solid").feature().remove(LOAD_TAG);
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").physics("solid").feature().remove(BODY_TAG);
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").physics("solid").feature().remove(RMS_TAG);
    } catch (Exception ignored) {
    }

    m.component("comp1").physics("solid").create(FIX_TAG, "Fixed", 2);
    m.component("comp1").physics("solid").feature(FIX_TAG).selection().named(SEL_FIX);

    m.component("comp1").physics("solid").create(LOAD_TAG, "BoundaryLoad", 2);
    m.component("comp1").physics("solid").feature(LOAD_TAG).selection().named(SEL_LOAD);
    m.component("comp1").physics("solid").feature(LOAD_TAG).set("F", new String[]{"0", "0", "-p_front"});

    m.component("comp1").physics("solid").create(BODY_TAG, "BodyLoad", 3);
    m.component("comp1").physics("solid").feature(BODY_TAG).selection().named(SEL_DOM);
    m.component("comp1").physics("solid").feature(BODY_TAG).set("F", new String[]{"0", "0", "force_density_z"});
    m.component("comp1").physics("solid").feature(BODY_TAG).set("FperVol", new String[]{"0", "0", "force_density_z"});

    m.component("comp1").physics("solid").create(RMS_TAG, "RigidMotionSuppression", 3);
    m.component("comp1").physics("solid").feature(RMS_TAG).selection().named(SEL_DOM);

    // Deactivate legacy load/support features to avoid conflicts.
    String[] legacy = new String[]{
        "fix1", "fixe_all", "fix_tail", "bndl1", "bndl_pr", "frontld", "body1", "bodyall", "rms1", "rmsd1", "bodyd1"
    };
    for (String t : legacy) {
      safeActivate(m, t, false);
    }
    safeActivate(m, FIX_TAG, true);
    safeActivate(m, LOAD_TAG, true);
    safeActivate(m, BODY_TAG, true);
    safeActivate(m, RMS_TAG, true);
  }

  private static void setMaterialForStudy(Model m, String study) {
    safeActivate(m, "lemm1", false);
    safeActivate(m, "hmm_nh", false);
    safeActivate(m, "hmm_og", false);
    safeActivate(m, "hmm_mr2", false);
    safeActivate(m, "hmm_mr5", false);

    if ("std1".equals(study)) {
      safeActivate(m, "lemm1", true);
      return;
    }
    if ("std_nh".equals(study)) {
      safeActivate(m, "hmm_nh", true);
      return;
    }
    if ("std_og".equals(study)) {
      safeActivate(m, "hmm_og", true);
      return;
    }
    if ("std_mr2".equals(study)) {
      safeActivate(m, "hmm_mr2", true);
      return;
    }
    safeActivate(m, "hmm_mr5", true);
  }

  private static double evalMaxVolume(Model m, String dset, String expr, String tag) {
    try {
      try {
        m.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        return r[0][0];
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static void rebuildMesh(Model m, List<String> logs) {
    m.mesh("mpart1").feature("imp1").set("source", "nastran");
    m.mesh("mpart1").feature("imp1").set("filename", BDF_VOL);
    m.mesh("mpart1").feature("imp1").set("createdom", "on");
    m.mesh("mpart1").feature("imp1").set("facepartition", "minimal");
    if (hasTag(m.mesh("mpart1").feature().tags(), "remf1")) {
      try {
        m.mesh("mpart1").feature().remove("remf1");
        logs.add("MESH_REMOVE|mpart1/remf1|ok=true");
      } catch (Exception e) {
        logs.add("MESH_REMOVE|mpart1/remf1|ok=false|err=" + safeMsg(e));
      }
    }

    m.mesh("mpart1").run("imp1");
    m.mesh("mpart1").run();
    logs.add("MESH_RUN|mpart1|ok=true");

    if (!hasTag(m.component("comp1").mesh("mesh1").feature().tags(), "impmsh")) {
      m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    }
    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
    try {
      m.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
    } catch (Exception ignored) {
    }
    m.component("comp1").mesh("mesh1").run("impmsh");
    m.component("comp1").mesh("mesh1").run();
    logs.add("MESH_RUN|comp1/mesh1|ok=true");
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
    logs.add("BDF_VOL|" + BDF_VOL);

    int b0 = countAll(m, 2, "tmp_bnd_before");
    int d0 = countAll(m, 3, "tmp_dom_before");
    logs.add("ENTITY_COUNTS_BEFORE|bnd=" + b0 + "|dom=" + d0);

    rebuildMesh(m, logs);

    int b1 = countAll(m, 2, "tmp_bnd_after_mesh");
    int d1 = countAll(m, 3, "tmp_dom_after_mesh");
    logs.add("ENTITY_COUNTS_AFTER_MESH|bnd=" + b1 + "|dom=" + d1);

    ensureSelections(m, logs);
    ensurePhysics(m, logs);

    // Force all static studies to use the repaired volumetric mesh.
    String[] studies = new String[]{"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};
    Map<String, String> dset = new LinkedHashMap<String, String>();
    dset.put("std1", "dset6");
    dset.put("std_nh", "dset1");
    dset.put("std_og", "dset2");
    dset.put("std_mr2", "dset3");
    dset.put("std_mr5", "dset4");
    dset.put("std_pr", "dset5");

    int good = 0;
    for (String std : studies) {
      if (!hasTag(m.study().tags(), std)) {
        logs.add("STUDY_SKIP|" + std + "|reason=missing");
        continue;
      }
      setMaterialForStudy(m, std);
      try {
        m.study(std).feature("stat").set("mesh", new String[][]{{"geom1", "mesh1"}});
      } catch (Exception ignored) {
      }
      try {
        m.study(std).run();
        logs.add("STUDY_RUN|" + std + "|ok=true");
      } catch (Exception e) {
        logs.add("STUDY_RUN|" + std + "|ok=false|err=" + safeMsg(e));
        continue;
      }

      String ds = dset.get(std);
      double vm = evalMaxVolume(m, ds, "solid.mises", "mxvm_" + std);
      double um = evalMaxVolume(m, ds, "sqrt(u^2+v^2+w^2)", "mxu_" + std);
      boolean ok = Double.isFinite(vm) && Double.isFinite(um) && Math.abs(vm) > 1e-12 && Math.abs(um) > 1e-15;
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
    }

    String backup = MPH + ".bak-" + ts();
    m.save(backup);
    logs.add("BACKUP|" + backup);
    m.save(MPH);
    logs.add("SAVED|" + MPH);

    logs.add("SUMMARY|finite_nonzero_studies=" + good + "|total_target_studies=" + studies.length);
    for (String line : logs) {
      System.out.println(line);
    }
  }
}
