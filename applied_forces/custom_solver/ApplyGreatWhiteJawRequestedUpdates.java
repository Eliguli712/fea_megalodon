import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

public class ApplyGreatWhiteJawRequestedUpdates {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";

  private static void p(String s) {
    System.out.println(s);
  }

  private static boolean hasPhysics(Model m, String comp, String phys) {
    try {
      m.component(comp).physics(phys);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasPhysicsFeature(Model m, String comp, String phys, String feat) {
    try {
      m.component(comp).physics(phys).feature(feat);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasStudy(Model m, String studyTag) {
    try {
      m.study(studyTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasStudyStep(Model m, String studyTag, String stepTag) {
    try {
      m.study(studyTag).feature(stepTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasResult(Model m, String tag) {
    try {
      m.result(tag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasResultFeature(Model m, String plotTag, String featTag) {
    try {
      m.result(plotTag).feature(featTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static void safeSetPF(Model m, String comp, String phys, String feat, String key, String value) {
    try {
      m.component(comp).physics(phys).feature(feat).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSetPFVec(Model m, String comp, String phys, String feat, String key, String[] value) {
    try {
      m.component(comp).physics(phys).feature(feat).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSetRF(Model m, String plotTag, String featTag, String key, String value) {
    try {
      m.result(plotTag).feature(featTag).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeActivatePF(Model m, String comp, String phys, String feat, boolean on) {
    try {
      m.component(comp).physics(phys).feature(feat).active(on);
    } catch (Exception ignored) {
    }
  }

  private static String[] datasetTags(Model m) {
    try {
      return m.result().dataset().tags();
    } catch (Exception e) {
      return new String[0];
    }
  }

  private static String newestDatasetTag(String[] before, String[] after) {
    Set<String> old = new LinkedHashSet<String>(Arrays.asList(before));
    for (String t : after) {
      if (!old.contains(t)) {
        return t;
      }
    }
    if (after.length > 0) {
      return after[after.length - 1];
    }
    return null;
  }

  private static void ensureSolidAndHyperelastic(Model m) {
    if (!hasPhysics(m, "comp1", "solid")) {
      m.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
      p("created physics solid");
    }
    m.component("comp1").physics("solid").label("Solid Mechanics");

    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_nh")) {
      m.component("comp1").physics("solid").create("hmm_nh", "HyperelasticModel", 3);
      p("created hmm_nh");
    }
    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_svk")) {
      m.component("comp1").physics("solid").create("hmm_svk", "HyperelasticModel", 3);
      p("created hmm_svk");
    }
    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_mr2")) {
      m.component("comp1").physics("solid").create("hmm_mr2", "HyperelasticModel", 3);
      p("created hmm_mr2");
    }
    if (!hasPhysicsFeature(m, "comp1", "solid", "hmm_mr5")) {
      m.component("comp1").physics("solid").create("hmm_mr5", "HyperelasticModel", 3);
      p("created hmm_mr5");
    }

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

    // Keep linear elastic off when hyperelastic studies are configured.
    safeActivatePF(m, "comp1", "solid", "lemm1", false);
  }

  private static void ensureStudy(Model m, String tag, String label) {
    if (!hasStudy(m, tag)) {
      m.study().create(tag);
      p("created study " + tag);
    }
    m.study(tag).label(label);
    if (!hasStudyStep(m, tag, "stat")) {
      m.study(tag).create("stat", "Stationary");
    }
    m.study(tag).feature("stat").activate("solid", true);
  }

  private static void setMaterialActivation(Model m, String activeTag) {
    String[] mats = new String[] {"hmm_nh", "hmm_svk", "hmm_mr2", "hmm_mr5"};
    for (String mt : mats) {
      if (hasPhysicsFeature(m, "comp1", "solid", mt)) {
        safeActivatePF(m, "comp1", "solid", mt, mt.equals(activeTag));
      }
    }
  }

  private static void ensureVonMisesPlot(Model m, String plotTag, String label, String datasetTag) {
    if (!hasResult(m, plotTag)) {
      m.result().create(plotTag, "PlotGroup3D");
    }
    m.result(plotTag).label(label);
    if (datasetTag != null && !datasetTag.isEmpty()) {
      try {
        m.result(plotTag).set("data", datasetTag);
      } catch (Exception ignored) {
      }
    }

    if (!hasResultFeature(m, plotTag, "surf1")) {
      m.result(plotTag).create("surf1", "Surface");
    }
    safeSetRF(m, plotTag, "surf1", "expr", "solid.mises");
    safeSetRF(m, plotTag, "surf1", "unit", "Pa");
    safeSetRF(m, plotTag, "surf1", "descr", "Von Mises stress");
  }

  public static Model run() {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load " + MPH, e);
    }

    p("loaded model: " + m.label());

    // Preserve full imported resolution and avoid smoothing/coarsening.
    try {
      m.component("comp1").mesh("mesh1").feature("join1").label("Lower Jaw Mesh");
      m.component("comp1").mesh("mesh1").feature("join2").label("Upper Jaw Mesh");
      p("updated mesh labels for upper/lower jaws");
    } catch (Exception e) {
      p("mesh label update warning: " + e.getMessage());
    }

    // Ensure boundary joins are built for clear jaw interfaces (no smoothing step added).
    try {
      m.component("comp1").mesh("mesh1").run("join1");
      m.component("comp1").mesh("mesh1").run("join2");
      m.component("comp1").mesh("mesh1").run("fin");
      p("rebuilt join boundaries without smoothing");
    } catch (Exception e) {
      p("mesh join rebuild warning: " + e.getMessage());
    }

    // Material parameters.
    m.param().set("mu_ref", "2.5e7[Pa]");
    m.param().descr("mu_ref", "Reference shear modulus for Neo-Hookean model.");
    m.param().set("kappa_bulk", "2.5e8[Pa]");
    m.param().descr("kappa_bulk", "Reference bulk modulus for near-incompressible response.");

    m.param().set("svk_E", "5.0e7[Pa]");
    m.param().descr("svk_E", "St. Venant-Kirchhoff Young's modulus.");
    m.param().set("svk_nu", "0.49");
    m.param().descr("svk_nu", "St. Venant-Kirchhoff Poisson ratio.");

    m.param().set("mr2_c10", "1.6e7[Pa]");
    m.param().descr("mr2_c10", "Mooney-Rivlin MR2 coefficient C10.");
    m.param().set("mr2_c01", "4.0e6[Pa]");
    m.param().descr("mr2_c01", "Mooney-Rivlin MR2 coefficient C01.");

    m.param().set("mr5_c10", "1.2e7[Pa]");
    m.param().descr("mr5_c10", "Mooney-Rivlin MR5 coefficient C10.");
    m.param().set("mr5_c01", "3.0e6[Pa]");
    m.param().descr("mr5_c01", "Mooney-Rivlin MR5 coefficient C01.");
    m.param().set("mr5_c20", "2.0e6[Pa]");
    m.param().descr("mr5_c20", "Mooney-Rivlin MR5 coefficient C20.");
    m.param().set("mr5_c11", "1.5e6[Pa]");
    m.param().descr("mr5_c11", "Mooney-Rivlin MR5 coefficient C11.");
    m.param().set("mr5_c02", "8.0e5[Pa]");
    m.param().descr("mr5_c02", "Mooney-Rivlin MR5 coefficient C02.");

    ensureSolidAndHyperelastic(m);

    ensureStudy(m, "std_mr2", "Von Mises Cloud std_mr2 (Static)");
    ensureStudy(m, "std_mr5", "Von Mises Cloud std_mr5 (Static)");

    String[] beforeMr2 = datasetTags(m);
    setMaterialActivation(m, "hmm_mr2");
    try {
      m.study("std_mr2").run();
      p("ran study std_mr2");
    } catch (Exception e) {
      p("std_mr2 run warning: " + e.getMessage());
    }
    String dsetMr2 = newestDatasetTag(beforeMr2, datasetTags(m));

    String[] beforeMr5 = datasetTags(m);
    setMaterialActivation(m, "hmm_mr5");
    try {
      m.study("std_mr5").run();
      p("ran study std_mr5");
    } catch (Exception e) {
      p("std_mr5 run warning: " + e.getMessage());
    }
    String dsetMr5 = newestDatasetTag(beforeMr5, datasetTags(m));

    ensureVonMisesPlot(m, "pg_vms_std_mr2", "Von Mises Cloud std_mr2", dsetMr2);
    ensureVonMisesPlot(m, "pg_vms_std_mr5", "Von Mises Cloud std_mr5", dsetMr5);

    // Keep Neo-Hookean visible by default while retaining SVK and MR models in the tree.
    setMaterialActivation(m, "hmm_nh");

    try {
      m.save(MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save updated model", e);
    }
    p("saved updated model: " + MPH);

    return m;
  }

  public static void main(String[] args) {
    run();
  }
}
