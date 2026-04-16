import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Locale;

public class BuildStaticDynamics {
  private static final String TEMPLATE_MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String OUTPUT_MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  private static final String[] ENTITY_NAMES = new String[]{
      "surface_mesh_smoothed",
      "tooth_surface_uncompressed",
      "tooth_surface_comsol_tet_vol"
  };
  private static final String[] PREFLIGHT_BDFS = new String[]{
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/surface_mesh_smoothed.bdf",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_comsol_tet_vol.bdf"
  };
  private static final String[] SOLVER_BDFS = new String[]{
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/surface_mesh_smoothed.bdf",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_comsol_tet_vol.bdf"
  };
  private static final String[] FORCE_DENSITY = new String[]{
      "5.0e4[N/m^3]",
      "5.0e4[N/m^3]",
      "5.0e4[N/m^3]"
  };

  private static final int M_MAX_DISP = 0;
  private static final int M_AVG_DISP = 1;
  private static final int M_MAX_STRESS = 2;
  private static final int M_AVG_STRESS = 3;
  private static final int M_MAX_STRAIN = 4;
  private static final int M_AVG_STRAIN = 5;
  private static final int M_MAX_ENERGY = 6;
  private static final int M_AVG_ENERGY = 7;
  private static final int M_VOLUME = 8;
  private static final int M_TOTAL_FORCE = 9;
  private static final int M_KEFF = 10;
  private static final int M_TANGENT = 11;
  private static final int M_COUNT = 12;

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

  private static void printKeyValue(String prefix, String[][] pairs) {
    StringBuilder sb = new StringBuilder(prefix);
    for (String[] pair : pairs) {
      sb.append("|").append(pair[0]).append("=").append(pair[1]);
    }
    System.out.println(sb);
  }

  private static boolean hasDataset(Model model, String tag) {
    try {
      for (String t : model.result().dataset().tags()) {
        if (tag.equals(t)) {
          return true;
        }
      }
    } catch (Exception ignored) {
    }
    return false;
  }

  private static void clearMesh1Features(Model model) {
    String[] tags = model.component("comp1").mesh("mesh1").feature().tags();
    for (String tag : tags) {
      if ("fin".equals(tag)) {
        continue;
      }
      try {
        model.component("comp1").mesh("mesh1").feature().remove(tag);
      } catch (Exception ignored) {
      }
    }
  }

  private static void clearMpartDeleteFeatures(Model model) {
    String[] tags = model.mesh("mpart1").feature().tags();
    for (String tag : tags) {
      if (tag.startsWith("dele")) {
        try {
          model.mesh("mpart1").feature().remove(tag);
        } catch (Exception ignored) {
        }
      }
    }
  }

  private static boolean preflightOpen(Model model, String entityName, String bdfPath) {
    try {
      model.mesh("mpart1").feature("imp1").set("source", "nastran");
      model.mesh("mpart1").feature("imp1").set("filename", bdfPath);
      model.mesh("mpart1").run("imp1");
      printKeyValue(
          "BDF_OPEN",
          new String[][]{
              {"entity", entityName},
              {"file", bdfPath},
              {"ok", "true"}
          }
      );
      return true;
    } catch (Exception e) {
      printKeyValue(
          "BDF_OPEN",
          new String[][]{
              {"entity", entityName},
              {"file", bdfPath},
              {"ok", "false"},
              {"error", safeMsg(e)}
          }
      );
      return false;
    }
  }

  private static void loadBdfForSolve(Model model, String bdfPath, boolean forceFreeTet) {
    clearMpartDeleteFeatures(model);
    model.mesh("mpart1").feature("imp1").set("source", "nastran");
    model.mesh("mpart1").feature("imp1").set("filename", bdfPath);
    model.mesh("mpart1").feature("imp1").set("createdom", "on");
    model.mesh("mpart1").feature("imp1").set("facepartition", "minimal");
    try {
      model.mesh("mpart1").feature("remf1").selection().all();
    } catch (Exception ignored) {
    }
    model.mesh("mpart1").run();

    clearMesh1Features(model);
    model.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
    model.component("comp1").mesh("mesh1").run("impmsh");

    if (forceFreeTet) {
      try {
        model.component("comp1").mesh("mesh1").feature().create("ftet1", "FreeTet");
        model.component("comp1").mesh("mesh1").feature("ftet1").selection().geom("geom1", 3);
        model.component("comp1").mesh("mesh1").feature("ftet1").selection().all();
        model.component("comp1").mesh("mesh1").run("ftet1");
      } catch (Exception e) {
        printKeyValue(
            "FREE_TET",
            new String[][]{
                {"file", bdfPath},
                {"ok", "false"},
                {"error", safeMsg(e)}
            }
        );
      }
    }

    model.component("comp1").mesh("mesh1").run("fin");
    try {
      model.study("std_mr5").feature("stat").set("mesh", new String[][]{{"geom1", "mesh1"}});
    } catch (Exception ignored) {
    }
  }

  private static void safeSet(Model model, String featureTag, String key, String value) {
    try {
      model.component("comp1").physics("solid").feature(featureTag).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSetVec(Model model, String featureTag, String key, String[] values) {
    try {
      model.component("comp1").physics("solid").feature(featureTag).set(key, values);
    } catch (Exception ignored) {
    }
  }

  private static void safeActivate(Model model, String featureTag, boolean active) {
    try {
      model.component("comp1").physics("solid").feature(featureTag).active(active);
    } catch (Exception ignored) {
    }
  }

  private static void removeSolidFeature(Model model, String tag) {
    try {
      model.component("comp1").physics("solid").feature().remove(tag);
    } catch (Exception ignored) {
    }
  }

  private static void configureMr5(Model model) {
    model.param().set("mr5_c10", "1.2e7[Pa]");
    model.param().set("mr5_c01", "3.0e6[Pa]");
    model.param().set("mr5_c20", "2.0e6[Pa]");
    model.param().set("mr5_c11", "1.5e6[Pa]");
    model.param().set("mr5_c02", "8.0e5[Pa]");
    model.param().set("kappa_bulk", "2.5e8[Pa]");
    model.param().set("mr5_formula_label", "1");
    model.param().descr(
        "mr5_formula_label",
        "W=C10*(I1bar-3)+C01*(I2bar-3)+C20*(I1bar-3)^2+C11*(I1bar-3)*(I2bar-3)+C02*(I2bar-3)^2+(kappa/2)*(J-1)^2"
    );
    model.param().set("spring_formula_label", "1");
    model.param().descr(
        "spring_formula_label",
        "F_total=f_body*Volume, k_eff=F_total/u_max, sigma_eq=solid.mises, epsilon_eq=eq_strain_from_grad_u"
    );

    safeActivate(model, "lemm1", false);
    safeActivate(model, "hmm_nh", false);
    safeActivate(model, "hmm_og", false);
    safeActivate(model, "hmm_mr2", false);
    safeActivate(model, "hmm_mr5", true);

    safeActivate(model, "fix1", false);
    safeActivate(model, "fixe_all", false);
    safeActivate(model, "bndl1", false);
    safeActivate(model, "bndl_pr", false);
    safeActivate(model, "bodyall", false);

    safeSet(model, "hmm_mr5", "MaterialModel", "MooneyRivlin5parameters");
    safeSet(model, "hmm_mr5", "Compressibility_MooneyRivlin", "NearlyIncompressible");
    safeSet(model, "hmm_mr5", "C10_mat", "userdef");
    safeSet(model, "hmm_mr5", "C10", "mr5_c10");
    safeSet(model, "hmm_mr5", "C01_mat", "userdef");
    safeSet(model, "hmm_mr5", "C01", "mr5_c01");
    safeSet(model, "hmm_mr5", "C20_mat", "userdef");
    safeSet(model, "hmm_mr5", "C20", "mr5_c20");
    safeSet(model, "hmm_mr5", "C11_mat", "userdef");
    safeSet(model, "hmm_mr5", "C11", "mr5_c11");
    safeSet(model, "hmm_mr5", "C02_mat", "userdef");
    safeSet(model, "hmm_mr5", "C02", "mr5_c02");
    safeSet(model, "hmm_mr5", "kappa", "kappa_bulk");

    // Keep first-order shape functions for numerical stability and memory on 19 GB machines.
    try {
      model.component("comp1").physics("solid").prop("ShapeProperty").set("order_displacement", "1");
    } catch (Exception ignored) {
    }
    try {
      model.component("comp1").physics("solid").prop("ShapeProperty").set("order_pressure", "1");
    } catch (Exception ignored) {
    }
    try {
      model.component("comp1").physics("solid").prop("ShapeProperty").set("displacementOrder", "linear");
    } catch (Exception ignored) {
    }
    try {
      model.study("std_mr5").feature("stat").set("shapeorder", "linear");
    } catch (Exception ignored) {
    }
  }

  private static void ensureBodyForceAndRms(Model model, String forceDensityExpr) {
    removeSolidFeature(model, "rmsd1");
    removeSolidFeature(model, "bodyd1");

    model.component("comp1").physics("solid").create("rmsd1", "RigidMotionSuppression", 3);
    model.component("comp1").physics("solid").feature("rmsd1").selection().all();

    model.component("comp1").physics("solid").create("bodyd1", "BodyLoad", 3);
    model.component("comp1").physics("solid").feature("bodyd1").selection().all();
    safeSetVec(model, "bodyd1", "F", new String[]{"0", "0", forceDensityExpr});
    safeSetVec(model, "bodyd1", "FperVol", new String[]{"0", "0", forceDensityExpr});
  }

  private static double evalOnce(
      Model model,
      String tag,
      String type,
      String expr,
      String dataTag
  ) {
    try {
      try {
        model.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      model.result().numerical().create(tag, type);
      model.result().numerical(tag).set("expr", new String[]{expr});
      model.result().numerical(tag).selection().all();
      if (dataTag != null && hasDataset(model, dataTag)) {
        model.result().numerical(tag).set("data", dataTag);
      }
      model.result().numerical(tag).setResult();
      double[][] r = model.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        return r[0][0];
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static double evalWithFallback(Model model, String tag, String type, String[] exprs) {
    for (String expr : exprs) {
      double v = evalOnce(model, tag, type, expr, "dset4");
      if (Double.isFinite(v)) {
        return v;
      }
      v = evalOnce(model, tag, type, expr, null);
      if (Double.isFinite(v)) {
        return v;
      }
    }
    return Double.NaN;
  }

  private static void setMetricParam(Model model, String name, double value, String unit, String descr) {
    if (!Double.isFinite(value)) {
      return;
    }
    String expr = String.format(Locale.US, "%.12e%s", value, unit == null ? "" : unit);
    model.param().set(name, expr);
    if (descr != null && !descr.isEmpty()) {
      model.param().descr(name, descr);
    }
  }

  private static void initNaN(double[] values) {
    for (int i = 0; i < values.length; i++) {
      values[i] = Double.NaN;
    }
  }

  private static double parseForceDensityBase(String forceDensityExpr) {
    try {
      int idx = forceDensityExpr.indexOf('[');
      String s = idx >= 0 ? forceDensityExpr.substring(0, idx) : forceDensityExpr;
      return Double.parseDouble(s.trim());
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static boolean metricsFiniteForConvergence(double[] metrics) {
    return Double.isFinite(metrics[M_MAX_STRAIN])
        && metrics[M_MAX_STRAIN] > 1e-16
        && Double.isFinite(metrics[M_MAX_STRESS])
        && Double.isFinite(metrics[M_TANGENT])
        && metrics[M_TANGENT] > 0.0;
  }

  private static boolean runEntity(
      Model model,
      int entityIndex,
      double[] outMetrics,
      double[] outForceScale,
      String[] outError
  ) {
    initNaN(outMetrics);
    outForceScale[0] = Double.NaN;
    outError[0] = "";

    String entityName = ENTITY_NAMES[entityIndex];
    String solverBdf = SOLVER_BDFS[entityIndex];
    String forceDensityExpr = FORCE_DENSITY[entityIndex];

    boolean forceFreeTet = false;
    try {
      loadBdfForSolve(model, solverBdf, forceFreeTet);
      configureMr5(model);
    } catch (Exception e) {
      outError[0] = safeMsg(e);
      return false;
    }

    final String eqStrainExpr =
        "sqrt((d(u,x))^2+(d(v,y))^2+(d(w,z))^2"
            + "+0.5*(d(u,y)+d(v,x))^2"
            + "+0.5*(d(u,z)+d(w,x))^2"
            + "+0.5*(d(v,z)+d(w,y))^2)";
    double baseForceDensity = parseForceDensityBase(forceDensityExpr);
    double[] scales = new double[]{1.0, 0.5, 0.2};

    for (double scale : scales) {
      String scaledForceExpr = String.format(Locale.US, "(%s)*%.6g", forceDensityExpr, scale);
      model.param().set("force_density_z", scaledForceExpr);
      model.param().descr("force_density_z", "Domain body force density in +z.");

      try {
        ensureBodyForceAndRms(model, "force_density_z");
        model.study("std_mr5").run();
      } catch (Exception e) {
        outError[0] = safeMsg(e);
        continue;
      }

      outMetrics[M_MAX_DISP] = evalWithFallback(
          model, "mxu_" + entityName, "MaxVolume", new String[]{"sqrt(u^2+v^2+w^2)"}
      );
      outMetrics[M_AVG_DISP] = evalWithFallback(
          model, "avu_" + entityName, "AvVolume", new String[]{"sqrt(u^2+v^2+w^2)"}
      );
      outMetrics[M_MAX_STRESS] = evalWithFallback(
          model, "mxs_" + entityName, "MaxVolume", new String[]{"solid.mises"}
      );
      outMetrics[M_AVG_STRESS] = evalWithFallback(
          model, "avs_" + entityName, "AvVolume", new String[]{"solid.mises"}
      );
      outMetrics[M_MAX_STRAIN] = evalWithFallback(
          model, "mxe_" + entityName, "MaxVolume", new String[]{eqStrainExpr, "abs(solid.eel11)"}
      );
      outMetrics[M_AVG_STRAIN] = evalWithFallback(
          model, "ave_" + entityName, "AvVolume", new String[]{eqStrainExpr, "abs(solid.eel11)"}
      );
      outMetrics[M_MAX_ENERGY] = evalWithFallback(
          model, "mxw_" + entityName, "MaxVolume", new String[]{"solid.Ws"}
      );
      outMetrics[M_AVG_ENERGY] = evalWithFallback(
          model, "avw_" + entityName, "AvVolume", new String[]{"solid.Ws"}
      );
      outMetrics[M_VOLUME] = evalWithFallback(
          model, "intv_" + entityName, "IntVolume", new String[]{"1"}
      );

      if (Double.isFinite(baseForceDensity) && Double.isFinite(outMetrics[M_VOLUME])) {
        outMetrics[M_TOTAL_FORCE] = baseForceDensity * scale * outMetrics[M_VOLUME];
      }
      if (Double.isFinite(outMetrics[M_TOTAL_FORCE]) && Double.isFinite(outMetrics[M_MAX_DISP])
          && Math.abs(outMetrics[M_MAX_DISP]) > 1e-16) {
        outMetrics[M_KEFF] = outMetrics[M_TOTAL_FORCE] / outMetrics[M_MAX_DISP];
      }
      if (Double.isFinite(outMetrics[M_MAX_STRESS]) && Double.isFinite(outMetrics[M_MAX_STRAIN])
          && Math.abs(outMetrics[M_MAX_STRAIN]) > 1e-16) {
        outMetrics[M_TANGENT] = outMetrics[M_MAX_STRESS] / outMetrics[M_MAX_STRAIN];
      }

      printKeyValue(
          "METRIC_PASS",
          new String[][]{
              {"entity", entityName},
              {"force_scale", String.format(Locale.US, "%.6g", scale)},
              {"max_strain", String.format(Locale.US, "%.9e", outMetrics[M_MAX_STRAIN])},
              {"tangent_modulus", String.format(Locale.US, "%.9e", outMetrics[M_TANGENT])},
              {"max_stress", String.format(Locale.US, "%.9e", outMetrics[M_MAX_STRESS])},
              {"max_disp", String.format(Locale.US, "%.9e", outMetrics[M_MAX_DISP])},
              {"volume", String.format(Locale.US, "%.9e", outMetrics[M_VOLUME])}
          }
      );

      if (metricsFiniteForConvergence(outMetrics)) {
        outForceScale[0] = scale;
        setMetricParam(model, entityName + "_max_disp", outMetrics[M_MAX_DISP], "[m]", "Maximum displacement magnitude.");
        setMetricParam(model, entityName + "_avg_disp", outMetrics[M_AVG_DISP], "[m]", "Average displacement magnitude.");
        setMetricParam(model, entityName + "_max_stress", outMetrics[M_MAX_STRESS], "[Pa]", "Maximum equivalent stress (solid.mises).");
        setMetricParam(model, entityName + "_avg_stress", outMetrics[M_AVG_STRESS], "[Pa]", "Average equivalent stress (solid.mises).");
        setMetricParam(model, entityName + "_max_strain", outMetrics[M_MAX_STRAIN], "", "Equivalent strain from displacement gradients.");
        setMetricParam(model, entityName + "_avg_strain", outMetrics[M_AVG_STRAIN], "", "Average equivalent strain from displacement gradients.");
        setMetricParam(model, entityName + "_max_energy_density", outMetrics[M_MAX_ENERGY], "[Pa]", "Maximum strain energy density (solid.Ws).");
        setMetricParam(model, entityName + "_avg_energy_density", outMetrics[M_AVG_ENERGY], "[Pa]", "Average strain energy density (solid.Ws).");
        setMetricParam(model, entityName + "_volume", outMetrics[M_VOLUME], "[m^3]", "Meshed domain volume used in the static solve.");
        setMetricParam(model, entityName + "_total_force", outMetrics[M_TOTAL_FORCE], "[N]", "Total applied static force: force_density_z * volume.");
        setMetricParam(model, entityName + "_k_eff", outMetrics[M_KEFF], "[N/m]", "Effective spring constant: total_force / max_disp.");
        setMetricParam(model, entityName + "_tangent_modulus", outMetrics[M_TANGENT], "[Pa]", "Spring-style tangent modulus: max_stress / max_strain.");
        return true;
      }
    }

    if (outError[0] == null || outError[0].isEmpty()) {
      outError[0] = "No finite converged max_strain/tangent_modulus across force scales.";
    }
    return false;
  }

  private static String fmt(double value) {
    if (!Double.isFinite(value)) {
      return "NaN";
    }
    return String.format(Locale.US, "%.6e", value);
  }

  private static String buildReport(
      boolean[] solved,
      double[] forceScale,
      String[] errors,
      double[][] metrics
  ) {
    StringBuilder sb = new StringBuilder();
    sb.append("# Static Dynamics (Mooney-Rivlin MR5, BDF-first feed)\n\n");
    sb.append("- Preflight opens are run first on: `surface_mesh_smoothed.bdf`, `tooth_surface_uncompressed.bdf`, `tooth_surface_comsol_tet_vol.bdf`.\n");
    sb.append("- Solver feeds use the same three files directly (no alternate compressed/comparison BDF substitution).\n");
    sb.append("- Equivalent strain used: `sqrt((d(u,x))^2+(d(v,y))^2+(d(w,z))^2+0.5*(d(u,y)+d(v,x))^2+0.5*(d(u,z)+d(w,x))^2+0.5*(d(v,z)+d(w,y))^2)`.\n\n");
    sb.append("| Entity | Solved | force scale | max_disp (m) | max_stress (Pa) | max_strain | tangent_modulus (Pa) |\n");
    sb.append("|---|---:|---:|---:|---:|---:|---:|\n");
    for (int i = 0; i < ENTITY_NAMES.length; i++) {
      sb.append("| ").append(ENTITY_NAMES[i]).append(" | ").append(solved[i] ? "yes" : "no").append(" | ")
          .append(fmt(forceScale[i])).append(" | ")
          .append(fmt(metrics[i][M_MAX_DISP])).append(" | ")
          .append(fmt(metrics[i][M_MAX_STRESS])).append(" | ")
          .append(fmt(metrics[i][M_MAX_STRAIN])).append(" | ")
          .append(fmt(metrics[i][M_TANGENT])).append(" |\n");
      if (!solved[i] && errors[i] != null && !errors[i].isEmpty()) {
        sb.append("\n> ").append(ENTITY_NAMES[i]).append(" error: ").append(errors[i]).append("\n\n");
      }
    }
    return sb.toString();
  }

  public static void main(String[] args) {
    Model model;
    try {
      model = ModelUtil.load("Model", TEMPLATE_MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load template MPH: " + TEMPLATE_MPH, e);
    }

    final int n = ENTITY_NAMES.length;
    boolean[] preflightOpenOk = new boolean[n];
    boolean[] solved = new boolean[n];
    double[] forceScale = new double[n];
    String[] errors = new String[n];
    double[][] metrics = new double[n][M_COUNT];

    for (int i = 0; i < n; i++) {
      preflightOpenOk[i] = preflightOpen(model, ENTITY_NAMES[i], PREFLIGHT_BDFS[i]);
    }

    for (int i = 0; i < n; i++) {
      double[] fs = new double[1];
      String[] err = new String[1];
      printKeyValue(
          "SOLVER_START",
          new String[][]{
              {"entity", ENTITY_NAMES[i]},
              {"solver_bdf", SOLVER_BDFS[i]},
              {"preflight_open_ok", Boolean.toString(preflightOpenOk[i])}
          }
      );
      solved[i] = runEntity(model, i, metrics[i], fs, err);
      forceScale[i] = fs[0];
      errors[i] = err[0];
      printKeyValue(
          "SOLVER_DONE",
          new String[][]{
              {"entity", ENTITY_NAMES[i]},
              {"ok", Boolean.toString(solved[i])},
              {"force_scale", fmt(forceScale[i])},
              {"error", errors[i] == null ? "" : errors[i]}
          }
      );
    }

    String report = buildReport(solved, forceScale, errors, metrics);
    model.comments(report);

    try {
      model.save(OUTPUT_MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save output MPH: " + OUTPUT_MPH, e);
    }
    System.out.println("Saved: " + OUTPUT_MPH);
  }
}
