import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class RunFrontStressSweepMR5 {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  // Keep sweep aligned with the repaired full-body mesh used by the main holocastic run.
  private static final String BDF = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_noperson2_fixinv1.bdf";
  private static final String DSET_MR5 = "dset4";

  private static final String TX = "(solid.sx*nx + solid.sxy*ny + solid.sxz*nz)";
  private static final String TY = "(solid.sxy*nx + solid.sy*ny + solid.syz*nz)";
  private static final String TZ = "(solid.sxz*nx + solid.syz*ny + solid.sz*nz)";
  private static final String TMAG = "sqrt((" + TX + ")^2 + (" + TY + ")^2 + (" + TZ + ")^2)";
  private static final String MDENS = "sqrt((y*(" + TZ + ")-z*(" + TY + "))^2 + (z*(" + TX + ")-x*(" + TZ + "))^2 + (x*(" + TY + ")-y*(" + TX + "))^2)";

  private static boolean hasSolidFeature(Model m, String tag) {
    try { m.component("comp1").physics("solid").feature(tag); return true; } catch (Exception e) { return false; }
  }

  private static void safeActivateSolid(Model m, String feat, boolean on) {
    try { m.component("comp1").physics("solid").feature(feat).active(on); } catch (Exception ignored) {}
  }

  private static void safeSetSolid(Model m, String feat, String key, String val) {
    try { m.component("comp1").physics("solid").feature(feat).set(key, val); } catch (Exception ignored) {}
  }

  private static void safeSetSolidVec(Model m, String feat, String key, String[] val) {
    try { m.component("comp1").physics("solid").feature(feat).set(key, val); } catch (Exception ignored) {}
  }

  private static void configureImport(Model m) {
    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception ignored) {}
    try { m.component("comp1").mesh("mesh1").feature().remove("fin"); } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", BDF);
    m.component("comp1").mesh("mesh1").run("impmsh");

    try { m.study("std_mr5").feature("stat").set("mesh", new String[][]{{"geom1","mesh1"}}); } catch (Exception ignored) {}
  }

  private static void configureSelections(Model m) {
    try { m.component("comp1").selection().remove("sel_front_stress"); } catch (Exception ignored) {}
    try { m.component("comp1").selection().remove("sel_trailing_edge"); } catch (Exception ignored) {}

    m.component("comp1").selection().create("sel_front_stress", "Box");
    m.component("comp1").selection("sel_front_stress").set("entitydim", 2);
    m.component("comp1").selection("sel_front_stress").set("xmin", -100.0);
    m.component("comp1").selection("sel_front_stress").set("xmax", 100.0);
    m.component("comp1").selection("sel_front_stress").set("ymin", -100.0);
    m.component("comp1").selection("sel_front_stress").set("ymax", 100.0);
    m.component("comp1").selection("sel_front_stress").set("zmin", 21.4);
    m.component("comp1").selection("sel_front_stress").set("zmax", 100.0);

    m.component("comp1").selection().create("sel_trailing_edge", "Box");
    m.component("comp1").selection("sel_trailing_edge").set("entitydim", 2);
    m.component("comp1").selection("sel_trailing_edge").set("xmin", -100.0);
    m.component("comp1").selection("sel_trailing_edge").set("xmax", 100.0);
    m.component("comp1").selection("sel_trailing_edge").set("ymin", -100.0);
    m.component("comp1").selection("sel_trailing_edge").set("ymax", 100.0);
    m.component("comp1").selection("sel_trailing_edge").set("zmin", -100.0);
    // Rear section of the main body (exclude detached low-z artifacts).
    m.component("comp1").selection("sel_trailing_edge").set("zmax", 19.85);

    int nf = m.component("comp1").selection("sel_front_stress").entities(2).length;
    int nt = m.component("comp1").selection("sel_trailing_edge").entities(2).length;
    System.out.println("front boundaries=" + nf + " trailing boundaries=" + nt);
  }

  private static void configurePhysics(Model m) {
    m.param().set("impact_velocity", "1[m/s]");
    m.param().set("front_force_N", "1000[N]");
    m.param().set("front_area_ref", "1[m^2]");
    m.param().set("front_stress", "front_force_N/front_area_ref");

    // Material toggles: MR5 only for elastic impact sweep.
    safeActivateSolid(m, "lemm1", false);
    safeActivateSolid(m, "hmm_nh", false);
    safeActivateSolid(m, "hmm_og", false);
    safeActivateSolid(m, "hmm_mr2", false);
    safeActivateSolid(m, "hmm_mr5", true);

    // Use trailing-edge support for the sweep; keep baseline all-edge constraints untouched outside this workflow.
    safeActivateSolid(m, "fix1", false);
    safeActivateSolid(m, "fixe_all", true);
    safeActivateSolid(m, "bodyall", false);
    safeActivateSolid(m, "bndl_pr", false);
    safeActivateSolid(m, "bndl1", false);

    if (!hasSolidFeature(m, "fix_tail")) {
      m.component("comp1").physics("solid").create("fix_tail", "Fixed", 2);
    }
    m.component("comp1").physics("solid").feature("fix_tail").selection().named("sel_trailing_edge");
    safeActivateSolid(m, "fix_tail", true);

    if (!hasSolidFeature(m, "frontld")) {
      m.component("comp1").physics("solid").create("frontld", "BoundaryLoad", 2);
    }
    m.component("comp1").physics("solid").feature("frontld").selection().named("sel_front_stress");
    safeSetSolid(m, "frontld", "forceType", "ForceArea");
    safeSetSolidVec(m, "frontld", "FperArea", new String[]{"0", "0", "front_stress"});
    safeActivateSolid(m, "frontld", true);
  }

  private static double evalSurface(Model m, String tag, String type, String expr, String selectionTag, String dset) {
    try {
      try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).selection().named(selectionTag);
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception e) {
      System.out.println("evalSurface failed " + tag + ": " + e.getMessage());
    }
    return Double.NaN;
  }

  private static double evalVolume(Model m, String tag, String type, String expr, String dset) {
    try {
      try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception e) {
      System.out.println("evalVolume failed " + tag + ": " + e.getMessage());
    }
    return Double.NaN;
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException(e); }

    configureImport(m);
    configureSelections(m);
    configurePhysics(m);

    // Compute front area once for force->stress conversion.
    double frontArea = evalSurface(m, "front_area_eval", "IntSurface", "1", "sel_front_stress", DSET_MR5);
    if (Double.isNaN(frontArea) || frontArea <= 0) frontArea = 1.0;
    m.param().set("front_area_ref", frontArea + "[m^2]");
    System.out.println("front_area_ref=" + frontArea + " m^2");

    double[] frontForces = new double[]{500, 1000, 1500, 2000, 2500, 3000, 3500, 4000};

    System.out.println("METRIC_HEADER,front_force_N,max_trailing_edge_force_N,max_impact_Nm,max_von_mises_Pa,avg_von_mises_Pa,instantaneous_impact_Wm2");

    for (double fn : frontForces) {
      m.param().set("front_force_N", fn + "[N]");

      try { m.study("std_mr5").run(); }
      catch (Exception e) {
        System.out.println("std_mr5 failed at " + fn + " N: " + e.getMessage());
      }

      double trailingForceN = evalSurface(m, "int_tail_force", "IntSurface", TMAG, "sel_trailing_edge", DSET_MR5);
      double impactNm = evalSurface(m, "int_tail_impact", "IntSurface", MDENS, "sel_trailing_edge", DSET_MR5);
      double maxMises = evalVolume(m, "max_mises_mr5", "MaxVolume", "solid.mises", DSET_MR5);
      double avgMises = evalVolume(m, "avg_mises_mr5", "AvVolume", "solid.mises", DSET_MR5);
      double instantWm2 = evalSurface(m, "max_front_impact", "MaxSurface", "(" + TMAG + ")*impact_velocity", "sel_front_stress", DSET_MR5);

      System.out.println(String.format(java.util.Locale.US,
        "METRIC_ROW,%.6f,%.12f,%.12f,%.12f,%.12f,%.12f",
        fn, trailingForceN, impactNm, maxMises, avgMises, instantWm2));
    }

    try { m.save(MPH); }
    catch (IOException e) { throw new RuntimeException(e); }

    System.out.println("MR5 front-stress sweep complete.");
  }
}
