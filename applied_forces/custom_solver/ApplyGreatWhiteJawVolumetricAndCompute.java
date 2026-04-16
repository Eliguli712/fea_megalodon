import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

public class ApplyGreatWhiteJawVolumetricAndCompute {
  private static String INPUT_MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";
  private static String OUTPUT_MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";
  private static String VOL_BDF = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw_fullres_tet_vol.bdf";

  // Parsed from great_white_jaw_vol_tet_vol.bdf GRID cards.
  private static final double XMIN = -191.502;
  private static final double XMAX = 198.7291;
  private static final double YMIN = -256.541;
  private static final double YMAX = 256.5412;
  private static final double ZMIN = -97.5579;
  private static final double ZMAX = 104.7844;

  private static void p(String s) { System.out.println(s); }

  private static boolean hasPhysics(Model m, String comp, String phys) {
    try { m.component(comp).physics(phys); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasPhysicsFeature(Model m, String comp, String phys, String feat) {
    try { m.component(comp).physics(phys).feature(feat); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasStudy(Model m, String studyTag) {
    try { m.study(studyTag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasStudyStep(Model m, String studyTag, String stepTag) {
    try { m.study(studyTag).feature(stepTag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasResult(Model m, String tag) {
    try { m.result(tag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasResultFeature(Model m, String plotTag, String featTag) {
    try { m.result(plotTag).feature(featTag); return true; } catch (Exception e) { return false; }
  }

  private static void safeSetPF(Model m, String comp, String phys, String feat, String key, String value) {
    try { m.component(comp).physics(phys).feature(feat).set(key, value); } catch (Exception ignored) {}
  }

  private static void safeSetPFVec(Model m, String comp, String phys, String feat, String key, String[] value) {
    try { m.component(comp).physics(phys).feature(feat).set(key, value); } catch (Exception ignored) {}
  }

  private static void safeActivatePF(Model m, String comp, String phys, String feat, boolean on) {
    try { m.component(comp).physics(phys).feature(feat).active(on); } catch (Exception ignored) {}
  }

  private static void safeSetRF(Model m, String plotTag, String featTag, String key, String value) {
    try { m.result(plotTag).feature(featTag).set(key, value); } catch (Exception ignored) {}
  }

  private static String[] datasetTags(Model m) {
    try { return m.result().dataset().tags(); } catch (Exception e) { return new String[0]; }
  }

  private static String[] solutionTags(Model m) {
    try { return m.sol().tags(); } catch (Exception e) { return new String[0]; }
  }

  private static String newestDatasetTag(String[] before, String[] after) {
    Set<String> old = new LinkedHashSet<String>(Arrays.asList(before));
    for (String t : after) if (!old.contains(t)) return t;
    return null;
  }

  private static String datasetForSolution(Model m, String solTag) {
    if (solTag == null) return null;
    String[] dsets = datasetTags(m);
    for (String ds : dsets) {
      try {
        if ("Solution".equals(m.result().dataset(ds).getType())) {
          String s = m.result().dataset(ds).getString("solution");
          if (solTag.equals(s)) return ds;
        }
      } catch (Exception ignored) {}
    }
    return null;
  }

  private static void importVolumetricMesh(Model m) {
    try { m.component("comp1").mesh("mesh1").feature().remove("join1"); p("removed join1"); } catch (Exception ignored) {}
    try { m.component("comp1").mesh("mesh1").feature().remove("join2"); p("removed join2"); } catch (Exception ignored) {}

    // Recreate the import feature to avoid stale state from previous import mode.
    try { m.component("comp1").mesh("mesh1").feature().remove("imp1"); p("removed imp1"); } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh1").feature().create("imp1", "Import");

    // Important: do not set domelem/createdom here; those keys can switch source back to native.
    m.component("comp1").mesh("mesh1").feature("imp1").set("filename", VOL_BDF);
    m.component("comp1").mesh("mesh1").feature("imp1").set("source", "nastran");
    try { m.component("comp1").mesh("mesh1").feature("imp1").set("linearelem", "on"); } catch (Exception ignored) {}
    p("imp1 source=" + m.component("comp1").mesh("mesh1").feature("imp1").getString("source"));
    p("imp1 filename=" + m.component("comp1").mesh("mesh1").feature("imp1").getString("filename"));

    m.component("comp1").mesh("mesh1").run("imp1");
    try { m.component("comp1").mesh("mesh1").run("fin"); } catch (Exception ignored) {}
    p("imported volumetric CTETRA mesh from BDF");
  }

  private static void ensureSolidAndHyperelastic(Model m) {
    if (!hasPhysics(m, "comp1", "solid")) {
      m.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
      p("created solid physics");
    }
    m.component("comp1").physics("solid").label("Solid Mechanics");
    m.component("comp1").physics("solid").selection().all();

    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_nh")) m.component("comp1").physics("solid").create("hmm_nh", "HyperelasticModel", 3);
    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_svk")) m.component("comp1").physics("solid").create("hmm_svk", "HyperelasticModel", 3);
    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_mr2")) m.component("comp1").physics("solid").create("hmm_mr2", "HyperelasticModel", 3);
    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_mr5")) m.component("comp1").physics("solid").create("hmm_mr5", "HyperelasticModel", 3);

    m.component("comp1").physics("solid").feature("hmm_nh").selection().all();
    m.component("comp1").physics("solid").feature("hmm_svk").selection().all();
    m.component("comp1").physics("solid").feature("hmm_mr2").selection().all();
    m.component("comp1").physics("solid").feature("hmm_mr5").selection().all();

    m.component("comp1").physics("solid").feature("hmm_nh").label("Neo-Hookean Hyperelastic");
    m.component("comp1").physics("solid").feature("hmm_svk").label("St. Venant-Kirchhoff Hyperelastic");
    m.component("comp1").physics("solid").feature("hmm_mr2").label("Mooney-Rivlin MR2 Hyperelastic");
    m.component("comp1").physics("solid").feature("hmm_mr5").label("Mooney-Rivlin MR5 Hyperelastic");

    safeSetPF(m, "comp1", "solid", "hmm_nh", "MaterialModel", "NeoHookean");
    safeSetPF(m, "comp1", "solid", "hmm_nh", "Compressibility_NeoHookean", "CompressibleUncoupled");
    safeSetPF(m, "comp1", "solid", "hmm_nh", "G_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_nh", "G", "mu_ref");
    safeSetPF(m, "comp1", "solid", "hmm_nh", "K_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_nh", "K", "kappa_bulk");

    safeSetPF(m, "comp1", "solid", "hmm_svk", "MaterialModel", "SaintVenantKirchhoff");
    safeSetPF(m, "comp1", "solid", "hmm_svk", "Compressibility_SaintVenantKirchhoff", "Compressible");
    safeSetPF(m, "comp1", "solid", "hmm_svk", "E_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_svk", "E", "svk_E");
    safeSetPF(m, "comp1", "solid", "hmm_svk", "nu_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_svk", "nu", "svk_nu");

    safeSetPF(m, "comp1", "solid", "hmm_mr2", "MaterialModel", "MooneyRivlin");
    safeSetPF(m, "comp1", "solid", "hmm_mr2", "Compressibility_MooneyRivlin", "NearlyIncompressible");
    safeSetPF(m, "comp1", "solid", "hmm_mr2", "C10_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_mr2", "C10", "mr2_c10");
    safeSetPF(m, "comp1", "solid", "hmm_mr2", "C01_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_mr2", "C01", "mr2_c01");
    safeSetPF(m, "comp1", "solid", "hmm_mr2", "kappa", "kappa_bulk");

    safeSetPF(m, "comp1", "solid", "hmm_mr5", "MaterialModel", "MooneyRivlin5parameters");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "Compressibility_MooneyRivlin", "NearlyIncompressible");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C10_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C10", "mr5_c10");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C01_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C01", "mr5_c01");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C20_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C20", "mr5_c20");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C11_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C11", "mr5_c11");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C02_mat", "userdef");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "C02", "mr5_c02");
    safeSetPF(m, "comp1", "solid", "hmm_mr5", "kappa", "kappa_bulk");

    // lemm1 is mandatory in this interface and cannot be disabled; clear its domain selection.
    if (hasPhysicsFeature(m, "comp1", "solid", "lemm1")) {
      try {
        m.component("comp1").physics("solid").feature("lemm1").selection().set(new int[]{});
        p("cleared lemm1 domain selection");
      } catch (Exception e) {
        p("could not clear lemm1 selection: " + e.getMessage());
      }
      // Keep fallback values defined to avoid unresolved symbols if COMSOL evaluates feature checks.
      safeSetPF(m, "comp1", "solid", "lemm1", "E_mat", "userdef");
      safeSetPF(m, "comp1", "solid", "lemm1", "E", "svk_E");
      safeSetPF(m, "comp1", "solid", "lemm1", "nu_mat", "userdef");
      safeSetPF(m, "comp1", "solid", "lemm1", "nu", "svk_nu");
    }
  }

  private static int selectionBoundaryCount(Model m, String selTag) {
    try {
      int[] b = m.component("comp1").selection(selTag).entities(2);
      return b == null ? 0 : b.length;
    } catch (Exception e) {
      return 0;
    }
  }

  private static void configureBoundarySelectionsAndLoads(Model m) {
    m.param().set("x_min", XMIN + "[m]");
    m.param().set("x_max", XMAX + "[m]");
    m.param().set("y_min", YMIN + "[m]");
    m.param().set("y_max", YMAX + "[m]");
    m.param().set("z_min", ZMIN + "[m]");
    m.param().set("z_max", ZMAX + "[m]");
    m.param().set("jaw_load", "1.0e4[Pa]");
    m.param().set("y_fix_max", "y_min+0.12*(y_max-y_min)");
    m.param().set("y_load_min", "y_max-0.12*(y_max-y_min)");
    m.param().set("z_fix_max", "z_min+0.12*(z_max-z_min)");
    m.param().set("z_load_min", "z_max-0.12*(z_max-z_min)");

    // Primary split by Y.
    try { m.component("comp1").selection().create("sel_fix_lower", "Box"); } catch (Exception ignored) {}
    m.component("comp1").selection("sel_fix_lower").label("Lower Region Fix (Y-min)");
    m.component("comp1").selection("sel_fix_lower").set("entitydim", "2");
    m.component("comp1").selection("sel_fix_lower").set("xmin", "x_min");
    m.component("comp1").selection("sel_fix_lower").set("xmax", "x_max");
    m.component("comp1").selection("sel_fix_lower").set("ymin", "y_min");
    m.component("comp1").selection("sel_fix_lower").set("ymax", "y_fix_max");
    m.component("comp1").selection("sel_fix_lower").set("zmin", "z_min");
    m.component("comp1").selection("sel_fix_lower").set("zmax", "z_max");

    try { m.component("comp1").selection().create("sel_load_upper", "Box"); } catch (Exception ignored) {}
    m.component("comp1").selection("sel_load_upper").label("Upper Region Load (Y-max)");
    m.component("comp1").selection("sel_load_upper").set("entitydim", "2");
    m.component("comp1").selection("sel_load_upper").set("xmin", "x_min");
    m.component("comp1").selection("sel_load_upper").set("xmax", "x_max");
    m.component("comp1").selection("sel_load_upper").set("ymin", "y_load_min");
    m.component("comp1").selection("sel_load_upper").set("ymax", "y_max");
    m.component("comp1").selection("sel_load_upper").set("zmin", "z_min");
    m.component("comp1").selection("sel_load_upper").set("zmax", "z_max");

    // Fallback split by Z if Y bands are empty.
    if (selectionBoundaryCount(m, "sel_fix_lower") == 0 || selectionBoundaryCount(m, "sel_load_upper") == 0) {
      m.component("comp1").selection("sel_fix_lower").set("ymin", "y_min");
      m.component("comp1").selection("sel_fix_lower").set("ymax", "y_max");
      m.component("comp1").selection("sel_fix_lower").set("zmin", "z_min");
      m.component("comp1").selection("sel_fix_lower").set("zmax", "z_fix_max");

      m.component("comp1").selection("sel_load_upper").set("ymin", "y_min");
      m.component("comp1").selection("sel_load_upper").set("ymax", "y_max");
      m.component("comp1").selection("sel_load_upper").set("zmin", "z_load_min");
      m.component("comp1").selection("sel_load_upper").set("zmax", "z_max");
    }

    int nFix = selectionBoundaryCount(m, "sel_fix_lower");
    int nLoad = selectionBoundaryCount(m, "sel_load_upper");
    p("boundary selection counts: fix=" + nFix + " load=" + nLoad);

    if (!hasPhysicsFeature(m, "comp1", "solid", "fix1")) {
      m.component("comp1").physics("solid").create("fix1", "Fixed", 2);
    }
    m.component("comp1").physics("solid").feature("fix1").label("Lower Region Fixed");
    m.component("comp1").physics("solid").feature("fix1").selection().named("sel_fix_lower");

    if (!hasPhysicsFeature(m, "comp1", "solid", "bndl1")) {
      m.component("comp1").physics("solid").create("bndl1", "BoundaryLoad", 2);
    }
    m.component("comp1").physics("solid").feature("bndl1").label("Upper Region Static Load");
    m.component("comp1").physics("solid").feature("bndl1").selection().named("sel_load_upper");
    safeSetPF(m, "comp1", "solid", "bndl1", "forceType", "ForceArea");
    safeSetPF(m, "comp1", "solid", "bndl1", "LoadType", "ForceArea");
    safeSetPFVec(m, "comp1", "solid", "bndl1", "FperArea", new String[]{"0", "-jaw_load", "0"});
    safeSetPFVec(m, "comp1", "solid", "bndl1", "Ftot", new String[]{"0", "-jaw_load", "0"});
    safeSetPFVec(m, "comp1", "solid", "bndl1", "force", new String[]{"0", "-jaw_load", "0"});
  }

  private static void ensureStudy(Model m, String tag, String label) {
    if (!hasStudy(m, tag)) m.study().create(tag);
    m.study(tag).label(label);
    if (!hasStudyStep(m, tag, "stat")) m.study(tag).create("stat", "Stationary");
    m.study(tag).feature("stat").activate("solid", true);
  }

  private static void setMaterialActivation(Model m, String activeTag) {
    String[] mats = new String[]{"hmm_nh", "hmm_svk", "hmm_mr2", "hmm_mr5"};
    for (String mt : mats) if (hasPhysicsFeature(m, "comp1", "solid", mt)) safeActivatePF(m, "comp1", "solid", mt, mt.equals(activeTag));
    safeActivatePF(m, "comp1", "solid", "bndl1", true);
    safeActivatePF(m, "comp1", "solid", "fix1", true);
  }

  private static double evalVolume(Model m, String tag, String type, String expr, String dataset) {
    try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
    try {
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).set("data", dataset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception e) {
      p("metric " + tag + " failed: " + e.getMessage());
    }
    return Double.NaN;
  }

  private static void ensureVonMisesPlot(Model m, String plotTag, String label, String dataset) {
    if (!hasResult(m, plotTag)) m.result().create(plotTag, "PlotGroup3D");
    m.result(plotTag).label(label);
    try { m.result(plotTag).set("data", dataset); } catch (Exception ignored) {}
    if (!hasResultFeature(m, plotTag, "surf1")) m.result(plotTag).create("surf1", "Surface");
    safeSetRF(m, plotTag, "surf1", "expr", "solid.mises");
    safeSetRF(m, plotTag, "surf1", "unit", "Pa");
    safeSetRF(m, plotTag, "surf1", "descr", "Von Mises stress");
  }

  public static Model run() {
    Model m;
    try { m = ModelUtil.load("Model", INPUT_MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to load " + INPUT_MPH, e); }

    importVolumetricMesh(m);

    // Material parameters
    m.param().set("mu_ref", "2.5e7[Pa]");
    m.param().set("kappa_bulk", "2.5e8[Pa]");
    m.param().set("svk_E", "5.0e7[Pa]");
    m.param().set("svk_nu", "0.49");
    m.param().set("mr2_c10", "1.6e7[Pa]");
    m.param().set("mr2_c01", "4.0e6[Pa]");
    m.param().set("mr5_c10", "1.2e7[Pa]");
    m.param().set("mr5_c01", "3.0e6[Pa]");
    m.param().set("mr5_c20", "2.0e6[Pa]");
    m.param().set("mr5_c11", "1.5e6[Pa]");
    m.param().set("mr5_c02", "8.0e5[Pa]");

    ensureSolidAndHyperelastic(m);
    configureBoundarySelectionsAndLoads(m);

    int domCount = 0;
    int bndCount = 0;
    try {
      m.component("comp1").physics("solid").selection().all();
      domCount = m.component("comp1").physics("solid").selection().entities(3).length;
      bndCount = m.component("comp1").physics("solid").selection().entities(2).length;
    } catch (Exception ignored) {}
    p("solid selection counts dom=" + domCount + " bnd=" + bndCount);
    if (domCount <= 0) throw new RuntimeException("Volumetric domain import failed: 0 domains");

    ensureStudy(m, "std_mr2", "Von Mises Cloud std_mr2 (Static Volumetric)");
    ensureStudy(m, "std_mr5", "Von Mises Cloud std_mr5 (Static Volumetric)");

    // Run MR2 study and collect metrics.
    String[] solBefore2 = solutionTags(m);
    setMaterialActivation(m, "hmm_mr2");
    m.study("std_mr2").run();
    String solMr2 = newestDatasetTag(solBefore2, solutionTags(m));
    if (solMr2 == null) solMr2 = "sol1";
    String dsMr2 = datasetForSolution(m, solMr2);
    if (dsMr2 == null) dsMr2 = "dset1";

    double mr2MaxMises = evalVolume(m, "mx_mr2_vms", "MaxVolume", "solid.mises", dsMr2);
    double mr2AvgMises = evalVolume(m, "av_mr2_vms", "AvVolume", "solid.mises", dsMr2);
    double mr2MaxDisp = evalVolume(m, "mx_mr2_u", "MaxVolume", "sqrt(u^2+v^2+w^2)", dsMr2);
    double mr2AvgDisp = evalVolume(m, "av_mr2_u", "AvVolume", "sqrt(u^2+v^2+w^2)", dsMr2);

    // Run MR5 study and collect metrics.
    String[] solBefore5 = solutionTags(m);
    setMaterialActivation(m, "hmm_mr5");
    m.study("std_mr5").run();
    String solMr5 = newestDatasetTag(solBefore5, solutionTags(m));
    if (solMr5 == null) solMr5 = "sol2";
    String dsMr5 = datasetForSolution(m, solMr5);
    if (dsMr5 == null) dsMr5 = "dset2";

    double mr5MaxMises = evalVolume(m, "mx_mr5_vms", "MaxVolume", "solid.mises", dsMr5);
    double mr5AvgMises = evalVolume(m, "av_mr5_vms", "AvVolume", "solid.mises", dsMr5);
    double mr5MaxDisp = evalVolume(m, "mx_mr5_u", "MaxVolume", "sqrt(u^2+v^2+w^2)", dsMr5);
    double mr5AvgDisp = evalVolume(m, "av_mr5_u", "AvVolume", "sqrt(u^2+v^2+w^2)", dsMr5);

    ensureVonMisesPlot(m, "pg_vms_std_mr2", "Von Mises Cloud std_mr2", dsMr2);
    ensureVonMisesPlot(m, "pg_vms_std_mr5", "Von Mises Cloud std_mr5", dsMr5);

    // Keep Neo-Hookean active as default tree state.
    setMaterialActivation(m, "hmm_nh");

    try { m.save(OUTPUT_MPH); p("saved updated model to " + OUTPUT_MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to save updated model", e); }

    p("METRIC|std_mr2|dataset=" + dsMr2 + "|max_mises=" + mr2MaxMises + "|avg_mises=" + mr2AvgMises + "|max_disp=" + mr2MaxDisp + "|avg_disp=" + mr2AvgDisp);
    p("METRIC|std_mr5|dataset=" + dsMr5 + "|max_mises=" + mr5MaxMises + "|avg_mises=" + mr5AvgMises + "|max_disp=" + mr5MaxDisp + "|avg_disp=" + mr5AvgDisp);

    return m;
  }

  public static void main(String[] args) {
    if (args != null && args.length >= 1 && args[0] != null && !args[0].isEmpty()) {
      VOL_BDF = args[0];
    }
    if (args != null && args.length >= 2 && args[1] != null && !args[1].isEmpty()) {
      INPUT_MPH = args[1];
    }
    if (args != null && args.length >= 3 && args[2] != null && !args[2].isEmpty()) {
      OUTPUT_MPH = args[2];
    }
    p("CONFIG|vol_bdf=" + VOL_BDF);
    p("CONFIG|input_mph=" + INPUT_MPH);
    p("CONFIG|output_mph=" + OUTPUT_MPH);
    run();
  }
}
