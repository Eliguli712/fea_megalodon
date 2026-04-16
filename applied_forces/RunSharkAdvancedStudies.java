import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class RunSharkAdvancedStudies {
  private static boolean physicsFeatureExists(Model model, String compTag, String physTag, String featTag) {
    try {
      model.component(compTag).physics(physTag).feature(featTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean studyExists(Model model, String studyTag) {
    try {
      model.study(studyTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean studyStepExists(Model model, String studyTag, String stepTag) {
    try {
      model.study(studyTag).feature(stepTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean resultExists(Model model, String tag) {
    try {
      model.result(tag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean resultFeatureExists(Model model, String plotTag, String featTag) {
    try {
      model.result(plotTag).feature(featTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static void safeSetPF(
      Model model,
      String compTag,
      String physTag,
      String featTag,
      String key,
      String value
  ) {
    try {
      model.component(compTag).physics(physTag).feature(featTag).set(key, value);
    } catch (Exception ignored) {
      // Keep going across COMSOL minor-version property differences.
    }
  }

  private static void safeSetPFVec(
      Model model,
      String compTag,
      String physTag,
      String featTag,
      String key,
      String[] value
  ) {
    try {
      model.component(compTag).physics(physTag).feature(featTag).set(key, value);
    } catch (Exception ignored) {
      // Keep going across COMSOL minor-version property differences.
    }
  }

  private static void safeActivatePF(
      Model model,
      String compTag,
      String physTag,
      String featTag,
      boolean on
  ) {
    try {
      model.component(compTag).physics(physTag).feature(featTag).active(on);
    } catch (Exception ignored) {
      // Keep going.
    }
  }

  private static void safeSetRF(
      Model model,
      String plotTag,
      String featTag,
      String key,
      String value
  ) {
    try {
      model.result(plotTag).feature(featTag).set(key, value);
    } catch (Exception ignored) {
      // Keep going across COMSOL minor-version property differences.
    }
  }

  private static void ensureStudy(Model model, String studyTag, String label) {
    if (!studyExists(model, studyTag)) {
      model.study().create(studyTag);
    }
    model.study(studyTag).label(label);
    if (!studyStepExists(model, studyTag, "stat")) {
      model.study(studyTag).create("stat", "Stationary");
    }
    model.study(studyTag).feature("stat").activate("solid", true);
  }

  private static void configureHyperelastic(Model model, String tag, String label) {
    if (!physicsFeatureExists(model, "comp1", "solid", tag)) {
      model.component("comp1").physics("solid").create(tag, "HyperelasticModel", 3);
    }

    model.component("comp1").physics("solid").feature(tag).label(label);
    safeSetPF(model, "comp1", "solid", tag, "VolumetricEnergyUncoupled", "polynomial");
    safeSetPF(model, "comp1", "solid", tag, "energySamplingPotential", "hyperelastic");
    safeSetPF(model, "comp1", "solid", tag, "MixedFormulation", "none");
    safeSetPF(model, "comp1", "solid", tag, "IsotropicOption", "Lame");
    safeSetPF(model, "comp1", "solid", tag, "K_mat", "userdef");
    safeSetPF(model, "comp1", "solid", tag, "K", "kappa_bulk");
    safeSetPF(model, "comp1", "solid", tag, "G_mat", "userdef");
    safeSetPF(model, "comp1", "solid", tag, "G", "mu_ref");
    safeSetPF(model, "comp1", "solid", tag, "Eequ", "1.0[GPa]");
    safeSetPF(model, "comp1", "solid", tag, "Gequ", "mu_ref");
  }

  private static void setCaseActivation(
      Model model,
      String activeMaterialTag,
      boolean usePressureLoad
  ) {
    String[] mats = new String[] {"hmm_nh", "hmm_og", "hmm_mr2", "hmm_mr5"};
    for (String mt : mats) {
      if (physicsFeatureExists(model, "comp1", "solid", mt)) {
        safeActivatePF(model, "comp1", "solid", mt, mt.equals(activeMaterialTag));
      }
    }

    if (physicsFeatureExists(model, "comp1", "solid", "lemm1")) {
      safeActivatePF(model, "comp1", "solid", "lemm1", false);
    }

    if (physicsFeatureExists(model, "comp1", "solid", "bndl1")) {
      safeActivatePF(model, "comp1", "solid", "bndl1", !usePressureLoad);
    }

    if (physicsFeatureExists(model, "comp1", "solid", "bndl_pr")) {
      safeActivatePF(model, "comp1", "solid", "bndl_pr", usePressureLoad);
    }
  }

  public static Model run() {
    String mphPath = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";

    Model model;
    try {
      model = ModelUtil.load("Model", mphPath);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load MPH file: " + mphPath, e);
    }

    model.param().set("mu_ref", "2.5e7[Pa]");
    model.param().descr("mu_ref", "Reference shear modulus for hyperelastic fits.");
    model.param().set("kappa_bulk", "2.5e8[Pa]");
    model.param().descr("kappa_bulk", "Reference bulk modulus for near-incompressible response.");

    model.param().set("ogden_mu1", "2.2e7[Pa]");
    model.param().descr("ogden_mu1", "Ogden first shear coefficient.");
    model.param().set("ogden_alpha1", "1.3");
    model.param().descr("ogden_alpha1", "Ogden first exponent.");

    model.param().set("mr2_c10", "1.6e7[Pa]");
    model.param().descr("mr2_c10", "Mooney-Rivlin MR2 coefficient C10.");
    model.param().set("mr2_c01", "4.0e6[Pa]");
    model.param().descr("mr2_c01", "Mooney-Rivlin MR2 coefficient C01.");

    model.param().set("mr5_c10", "1.2e7[Pa]");
    model.param().descr("mr5_c10", "Mooney-Rivlin MR5 coefficient C10.");
    model.param().set("mr5_c01", "3.0e6[Pa]");
    model.param().descr("mr5_c01", "Mooney-Rivlin MR5 coefficient C01.");
    model.param().set("mr5_c20", "2.0e6[Pa]");
    model.param().descr("mr5_c20", "Mooney-Rivlin MR5 coefficient C20.");
    model.param().set("mr5_c11", "1.5e6[Pa]");
    model.param().descr("mr5_c11", "Mooney-Rivlin MR5 coefficient C11.");
    model.param().set("mr5_c02", "8.0e5[Pa]");
    model.param().descr("mr5_c02", "Mooney-Rivlin MR5 coefficient C02.");

    if (!physicsFeatureExists(model, "comp1", "solid", "bndl1")) {
      model.component("comp1").physics("solid").create("bndl1", "BoundaryLoad", 2);
    }
    model.component("comp1").physics("solid").feature("bndl1").label("Snout thrust load (force per area)");
    try {
      model.component("comp1").physics("solid").feature("bndl1").selection().named("sel_snout");
    } catch (Exception ignored) {
      // Keep existing selection if named selection is unavailable.
    }
    safeSetPF(model, "comp1", "solid", "bndl1", "forceType", "ForceArea");
    safeSetPF(model, "comp1", "solid", "bndl1", "force_src", "userdef");
    safeSetPFVec(model, "comp1", "solid", "bndl1", "force", new String[] {"0", "0", "thrust_load"});

    if (!physicsFeatureExists(model, "comp1", "solid", "bndl_pr")) {
      model.component("comp1").physics("solid").create("bndl_pr", "BoundaryLoad", 2);
    }
    model.component("comp1").physics("solid").feature("bndl_pr").label("Snout pressure load");
    try {
      model.component("comp1").physics("solid").feature("bndl_pr").selection().named("sel_snout");
    } catch (Exception ignored) {
      // Keep existing selection if named selection is unavailable.
    }
    safeSetPF(model, "comp1", "solid", "bndl_pr", "forceType", "FollowerPressure");
    safeSetPF(model, "comp1", "solid", "bndl_pr", "pressure", "thrust_load");

    configureHyperelastic(model, "hmm_nh", "Neo-Hookean Hyperelastic");
    configureHyperelastic(model, "hmm_og", "Ogden Hyperelastic");
    configureHyperelastic(model, "hmm_mr2", "Mooney-Rivlin MR2 Hyperelastic");
    configureHyperelastic(model, "hmm_mr5", "Mooney-Rivlin MR5 Hyperelastic");

    safeSetPF(model, "comp1", "solid", "hmm_nh", "MaterialModel", "NeoHookean");
    safeSetPF(model, "comp1", "solid", "hmm_nh", "Compressibility_NeoHookean", "CompressibleUncoupled");
    safeSetPF(model, "comp1", "solid", "hmm_nh", "G_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_nh", "G", "mu_ref");
    safeSetPF(model, "comp1", "solid", "hmm_nh", "K_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_nh", "K", "kappa_bulk");
    safeSetPF(model, "comp1", "solid", "hmm_nh", "kappa", "kappa_bulk");

    safeSetPF(model, "comp1", "solid", "hmm_og", "MaterialModel", "Ogden");
    safeSetPF(model, "comp1", "solid", "hmm_og", "Compressibility_Ogden", "NearlyIncompressible");
    safeSetPF(model, "comp1", "solid", "hmm_og", "mup", "ogden_mu1");
    safeSetPF(model, "comp1", "solid", "hmm_og", "alphap", "ogden_alpha1");
    safeSetPF(model, "comp1", "solid", "hmm_og", "muk", "0[Pa]");
    safeSetPF(model, "comp1", "solid", "hmm_og", "alphak", "1");
    safeSetPF(model, "comp1", "solid", "hmm_og", "betak", "1");
    safeSetPF(model, "comp1", "solid", "hmm_og", "kappa", "kappa_bulk");

    safeSetPF(model, "comp1", "solid", "hmm_mr2", "MaterialModel", "MooneyRivlin");
    safeSetPF(model, "comp1", "solid", "hmm_mr2", "Compressibility_MooneyRivlin", "NearlyIncompressible");
    safeSetPF(model, "comp1", "solid", "hmm_mr2", "C10_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_mr2", "C10", "mr2_c10");
    safeSetPF(model, "comp1", "solid", "hmm_mr2", "C01_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_mr2", "C01", "mr2_c01");
    safeSetPF(model, "comp1", "solid", "hmm_mr2", "kappa", "kappa_bulk");

    safeSetPF(model, "comp1", "solid", "hmm_mr5", "MaterialModel", "MooneyRivlin5parameters");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "Compressibility_MooneyRivlin", "NearlyIncompressible");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C10_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C10", "mr5_c10");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C01_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C01", "mr5_c01");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C20_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C20", "mr5_c20");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C11_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C11", "mr5_c11");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C02_mat", "userdef");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "C02", "mr5_c02");
    safeSetPF(model, "comp1", "solid", "hmm_mr5", "kappa", "kappa_bulk");

    ensureStudy(model, "std_nh", "Neo-Hookean thrust study");
    ensureStudy(model, "std_og", "Ogden thrust study");
    ensureStudy(model, "std_mr2", "Mooney-Rivlin MR2 thrust study");
    ensureStudy(model, "std_mr5", "Mooney-Rivlin MR5 thrust study");
    ensureStudy(model, "std_pr", "Pressure analysis (MR5)");

    try {
      setCaseActivation(model, "hmm_nh", false);
      model.study("std_nh").run();
      System.out.println("Completed study: std_nh");
    } catch (Exception e) {
      System.out.println("Study std_nh failed: " + e.getMessage());
    }

    try {
      setCaseActivation(model, "hmm_og", false);
      model.study("std_og").run();
      System.out.println("Completed study: std_og");
    } catch (Exception e) {
      System.out.println("Study std_og failed: " + e.getMessage());
    }

    try {
      setCaseActivation(model, "hmm_mr2", false);
      model.study("std_mr2").run();
      System.out.println("Completed study: std_mr2");
    } catch (Exception e) {
      System.out.println("Study std_mr2 failed: " + e.getMessage());
    }

    try {
      setCaseActivation(model, "hmm_mr5", false);
      model.study("std_mr5").run();
      System.out.println("Completed study: std_mr5");
    } catch (Exception e) {
      System.out.println("Study std_mr5 failed: " + e.getMessage());
    }

    try {
      setCaseActivation(model, "hmm_mr5", true);
      model.study("std_pr").run();
      System.out.println("Completed study: std_pr");
    } catch (Exception e) {
      System.out.println("Study std_pr failed: " + e.getMessage());
    }

    if (!resultExists(model, "pg_vms")) {
      model.result().create("pg_vms", "PlotGroup3D");
    }
    model.result("pg_vms").label("Von Mises Stress Cloud");
    if (!resultFeatureExists(model, "pg_vms", "surf_vms")) {
      model.result("pg_vms").create("surf_vms", "Surface");
    }
    safeSetRF(model, "pg_vms", "surf_vms", "expr", "solid.mises");
    safeSetRF(model, "pg_vms", "surf_vms", "unit", "Pa");
    safeSetRF(model, "pg_vms", "surf_vms", "descr", "Von Mises stress");

    setCaseActivation(model, "hmm_mr5", false);

    try {
      model.save(mphPath);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save updated model: " + mphPath, e);
    }

    return model;
  }

  public static void main(String[] args) {
    run();
  }
}
