import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class RunPressureCheck {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String BDF = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_noperson2_fixinv1.bdf";

  private static boolean hasStudy(Model m, String tag) {
    try { m.study(tag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasSolidFeature(Model m, String tag) {
    try { m.component("comp1").physics("solid").feature(tag); return true; } catch (Exception e) { return false; }
  }

  private static void safeSetSolid(Model m, String feat, String key, String val) {
    try { m.component("comp1").physics("solid").feature(feat).set(key, val); } catch (Exception ignored) {}
  }

  private static void safeSetSolidVec(Model m, String feat, String key, String[] val) {
    try { m.component("comp1").physics("solid").feature(feat).set(key, val); } catch (Exception ignored) {}
  }

  private static void safeActivateSolid(Model m, String feat, boolean on) {
    try { m.component("comp1").physics("solid").feature(feat).active(on); } catch (Exception ignored) {}
  }

  private static void configureMesh(Model m) {
    try { m.geom("part1").feature().remove("tor1"); } catch (Exception ignored) {}
    try { m.geom("part1").inputParam().set("solid", "0"); } catch (Exception ignored) {}
    try { m.geom("part1").inputParam().set("endsolid", "0"); } catch (Exception ignored) {}

    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception ignored) {}
    try { m.component("comp1").mesh("mesh1").feature().remove("fin"); } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", BDF);
    m.component("comp1").mesh("mesh1").run("impmsh");

    if (hasStudy(m, "std_pr")) {
      try { m.study("std_pr").feature("stat").set("mesh", new String[][]{{"geom1", "mesh1"}}); } catch (Exception ignored) {}
      try { m.study("std_pr").feature("stat").set("plot", "off"); } catch (Exception ignored) {}
    }
  }

  private static void configurePhysics(Model m) {
    m.param().set("pressure_global", "2e3[Pa]");
    m.param().set("kappa_bulk", "2.5e8[Pa]");
    m.param().set("mu_ref", "2.5e7[Pa]");
    m.param().set("lambda_ref", "kappa_bulk-2*mu_ref/3");
    m.param().set("mr5_c10", "1.2e7[Pa]");
    m.param().set("mr5_c01", "3.0e6[Pa]");
    m.param().set("mr5_c20", "2.0e6[Pa]");
    m.param().set("mr5_c11", "1.5e6[Pa]");
    m.param().set("mr5_c02", "8.0e5[Pa]");

    m.component("comp1").physics("solid").selection().all();

    if (!hasSolidFeature(m, "fix1")) m.component("comp1").physics("solid").create("fix1", "Fixed", 2);
    m.component("comp1").physics("solid").feature("fix1").selection().all();
    safeActivateSolid(m, "fix1", true);

    if (!hasSolidFeature(m, "fixe_all")) m.component("comp1").physics("solid").create("fixe_all", "Fixed", 1);
    m.component("comp1").physics("solid").feature("fixe_all").selection().all();
    safeActivateSolid(m, "fixe_all", true);

    if (hasSolidFeature(m, "bndl_pr")) {
      m.component("comp1").physics("solid").feature("bndl_pr").selection().all();
      safeSetSolid(m, "bndl_pr", "forceType", "FollowerPressure");
      safeSetSolid(m, "bndl_pr", "pressure", "pressure_global");
    }

    safeActivateSolid(m, "bodyall", false);
    safeActivateSolid(m, "bndl1", false);
    safeActivateSolid(m, "rms1", false);

    safeActivateSolid(m, "lemm1", false);
    safeActivateSolid(m, "hmm_nh", false);
    safeActivateSolid(m, "hmm_og", false);
    safeActivateSolid(m, "hmm_mr2", false);
    safeActivateSolid(m, "hmm_mr5", true);

    for (String feat : new String[]{"hmm_mr5"}) {
      if (hasSolidFeature(m, feat)) {
        try { m.component("comp1").physics("solid").feature(feat).selection().all(); } catch (Exception ignored) {}
      }
    }

    safeSetSolid(m, "hmm_mr5", "MaterialModel", "MooneyRivlin5parameters");
    safeSetSolid(m, "hmm_mr5", "muLame_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "muLame", "mu_ref");
    safeSetSolid(m, "hmm_mr5", "lambLame_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "lambLame", "lambda_ref");
    safeSetSolid(m, "hmm_mr5", "K2_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "K2", "kappa_bulk");
    safeSetSolid(m, "hmm_mr5", "K3_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "K3", "0[Pa]");
    safeSetSolid(m, "hmm_mr5", "C10_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C10", "mr5_c10");
    safeSetSolid(m, "hmm_mr5", "C01_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C01", "mr5_c01");
    safeSetSolid(m, "hmm_mr5", "C20_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C20", "mr5_c20");
    safeSetSolid(m, "hmm_mr5", "C11_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C11", "mr5_c11");
    safeSetSolid(m, "hmm_mr5", "C02_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C02", "mr5_c02");
    safeSetSolid(m, "hmm_mr5", "kappa", "kappa_bulk");
  }

  private static double evalMaxMises(Model m, String dataset) {
    try {
      try { m.result().numerical().remove("mxchk_pr"); } catch (Exception ignored) {}
      m.result().numerical().create("mxchk_pr", "MaxVolume");
      m.result().numerical("mxchk_pr").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mxchk_pr").set("unit", new String[]{"Pa"});
      m.result().numerical("mxchk_pr").set("data", dataset);
      m.result().numerical("mxchk_pr").selection().all();
      m.result().numerical("mxchk_pr").setResult();
      double[][] r = m.result().numerical("mxchk_pr").getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception e) {
      System.out.println("eval failed: " + e.getMessage());
    }
    return Double.NaN;
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException("load failed", e); }

    configureMesh(m);
    configurePhysics(m);

    int d=0,b=0,e=0;
    try { d = m.component("comp1").physics("solid").selection().entities(3).length; } catch (Exception ignored) {}
    try { b = m.component("comp1").physics("solid").feature("fix1").selection().entities(2).length; } catch (Exception ignored) {}
    try { e = m.component("comp1").physics("solid").feature("fixe_all").selection().entities(1).length; } catch (Exception ignored) {}
    System.out.println("counts dom=" + d + " bnd=" + b + " edge=" + e);

    if (!hasStudy(m, "std_pr")) throw new RuntimeException("std_pr missing");
    try { m.study("std_pr").run(); }
    catch (Exception ex) { throw new RuntimeException("std_pr failed: " + ex.getMessage(), ex); }

    double mx = evalMaxMises(m, "dset5");
    System.out.println("std_pr max mises=" + mx);

    try { m.save(MPH); }
    catch (IOException ex) { throw new RuntimeException("save failed", ex); }

    System.out.println("RunPressureCheck done");
  }
}
