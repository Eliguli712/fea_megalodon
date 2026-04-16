import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

public class BuildStaticDynamicsHighResolution {
  private static final String TEMPLATE_MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";
  private static final String OUTPUT_MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics_high_resolution.mph";
  private static final String OUTPUT_MD =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics_high_resolution_report.md";

  private static final String[] ENTITY_NAMES = new String[]{
      "surface_mesh_smoothed",
      "tooth_surface_uncompressed",
      "tooth_surface_comsol_tet_vol"
  };

  // Explicitly preserved input files from user request (never modified by this script).
  private static final String[] INPUT_BDFS = new String[]{
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/surface_mesh_smoothed.bdf",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_comsol_tet_vol.bdf"
  };

  // Solve directly on exactly the three requested input BDF files.
  private static final String[] SOLVER_BDFS = INPUT_BDFS;

  private static final String[] MATERIALS = new String[]{"stvkirchhoff", "mr2", "mr5"};
  private static final String[] MODES = new String[]{"linear", "nonlinear"};

  private static final int M_MAX_DISP = 0;
  private static final int M_MAX_STRESS = 1;
  private static final int M_MAX_STRAIN = 2;
  private static final int M_TANGENT = 3;
  private static final int M_COUNT = 4;

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

  private static void printKV(String prefix, String[][] pairs) {
    StringBuilder sb = new StringBuilder(prefix);
    for (String[] pair : pairs) {
      sb.append("|").append(pair[0]).append("=").append(pair[1]);
    }
    System.out.println(sb);
  }

  private static boolean hasStudy(Model model, String studyTag) {
    return Arrays.asList(model.study().tags()).contains(studyTag);
  }

  private static boolean hasStudyFeature(Model model, String studyTag, String featTag) {
    try {
      return Arrays.asList(model.study(studyTag).feature().tags()).contains(featTag);
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
      printKV(
          "BDF_OPEN",
          new String[][]{
              {"entity", entityName},
              {"file", bdfPath},
              {"ok", "true"}
          }
      );
      return true;
    } catch (Exception e) {
      printKV(
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

  private static void loadBdfForSolve(Model model, String bdfPath) {
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
    model.component("comp1").mesh("mesh1").run("fin");
  }

  private static void safeActivate(Model model, String featureTag, boolean active) {
    try {
      model.component("comp1").physics("solid").feature(featureTag).active(active);
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

  private static void removeSolidFeature(Model model, String tag) {
    try {
      model.component("comp1").physics("solid").feature().remove(tag);
    } catch (Exception ignored) {
    }
  }

  private static void ensureBodyForceAndRms(Model model) {
    removeSolidFeature(model, "rmsd1");
    removeSolidFeature(model, "bodyd1");

    model.component("comp1").physics("solid").create("rmsd1", "RigidMotionSuppression", 3);
    model.component("comp1").physics("solid").feature("rmsd1").selection().all();

    model.component("comp1").physics("solid").create("bodyd1", "BodyLoad", 3);
    model.component("comp1").physics("solid").feature("bodyd1").selection().all();
    safeSetVec(model, "bodyd1", "F", new String[]{"0", "0", "force_density_z"});
    safeSetVec(model, "bodyd1", "FperVol", new String[]{"0", "0", "force_density_z"});
  }

  private static void configureMaterial(Model model, String materialKey) {
    model.param().set("force_density_z", "5.0e4[N/m^3]");
    model.param().set("kappa_bulk", "2.5e8[Pa]");
    model.param().set("stvk_E", "1.5e8[Pa]");
    model.param().set("stvk_nu", "0.30");
    model.param().set("mr2_c10", "1.6e7[Pa]");
    model.param().set("mr2_c01", "4.0e6[Pa]");
    model.param().set("mr5_c10", "1.2e7[Pa]");
    model.param().set("mr5_c01", "3.0e6[Pa]");
    model.param().set("mr5_c20", "2.0e6[Pa]");
    model.param().set("mr5_c11", "1.5e6[Pa]");
    model.param().set("mr5_c02", "8.0e5[Pa]");

    safeActivate(model, "lemm1", false);
    safeActivate(model, "hmm_nh", false);
    safeActivate(model, "hmm_og", false);
    safeActivate(model, "hmm_mr2", false);
    safeActivate(model, "hmm_mr5", false);

    if ("stvkirchhoff".equals(materialKey)) {
      safeActivate(model, "hmm_mr2", true);
      safeSet(model, "hmm_mr2", "MaterialModel", "SaintVenantKirchhoff");
      safeSet(model, "hmm_mr2", "E_mat", "userdef");
      safeSet(model, "hmm_mr2", "E", "stvk_E");
      safeSet(model, "hmm_mr2", "nu_mat", "userdef");
      safeSet(model, "hmm_mr2", "nu", "stvk_nu");
    } else if ("mr2".equals(materialKey)) {
      safeActivate(model, "hmm_mr2", true);
      safeSet(model, "hmm_mr2", "MaterialModel", "MooneyRivlin");
      safeSet(model, "hmm_mr2", "Compressibility_MooneyRivlin", "NearlyIncompressible");
      safeSet(model, "hmm_mr2", "C10_mat", "userdef");
      safeSet(model, "hmm_mr2", "C10", "mr2_c10");
      safeSet(model, "hmm_mr2", "C01_mat", "userdef");
      safeSet(model, "hmm_mr2", "C01", "mr2_c01");
      safeSet(model, "hmm_mr2", "kappa", "kappa_bulk");
    } else {
      safeActivate(model, "hmm_mr5", true);
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
    }

    try {
      model.component("comp1").physics("solid").prop("ShapeProperty").set("order_displacement", "1");
      model.component("comp1").physics("solid").prop("ShapeProperty").set("order_pressure", "1");
      model.component("comp1").physics("solid").prop("ShapeProperty").set("displacementOrder", "linear");
    } catch (Exception ignored) {
    }
  }

  private static String caseStudyTag(String materialKey, String modeKey) {
    return "std_" + materialKey + "_" + ("linear".equals(modeKey) ? "lin" : "nl");
  }

  private static void ensureCaseStudy(Model model, String studyTag, boolean geometricNonlinear) {
    if (!hasStudy(model, studyTag)) {
      model.study().create(studyTag);
    }
    if (!hasStudyFeature(model, studyTag, "stat")) {
      model.study(studyTag).create("stat", "Stationary");
    }
    model.study(studyTag).feature("stat").set("mesh", new String[][]{{"geom1", "mesh1"}});
    model.study(studyTag).feature("stat").set("geometricNonlinearity", geometricNonlinear ? "on" : "off");
    try {
      model.study(studyTag).feature("stat").set("shapeorder", "linear");
    } catch (Exception ignored) {
    }
  }

  private static void initNaN(double[] values) {
    for (int i = 0; i < values.length; i++) {
      values[i] = Double.NaN;
    }
  }

  private static double evalOnce(Model model, String tag, String type, String expr) {
    try {
      try {
        model.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      model.result().numerical().create(tag, type);
      model.result().numerical(tag).set("expr", new String[]{expr});
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

  private static double evalWithFallback(Model model, String tag, String type, String[] exprs) {
    for (String expr : exprs) {
      double v = evalOnce(model, tag, type, expr);
      if (Double.isFinite(v)) {
        return v;
      }
    }
    return Double.NaN;
  }

  private static String safeToken(String s) {
    return s.replaceAll("[^A-Za-z0-9_]", "_");
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

  private static boolean runCase(
      Model model,
      String entity,
      String material,
      String mode,
      double[] outMetrics,
      String[] outError
  ) {
    initNaN(outMetrics);
    outError[0] = "";

    boolean nonlinear = "nonlinear".equals(mode);
    String studyTag = caseStudyTag(material, mode);
    String token = safeToken(entity + "_" + material + "_" + mode);
    final String eqStrainExpr =
        "sqrt((d(u,x))^2+(d(v,y))^2+(d(w,z))^2"
            + "+0.5*(d(u,y)+d(v,x))^2"
            + "+0.5*(d(u,z)+d(w,x))^2"
            + "+0.5*(d(v,z)+d(w,y))^2)";

    try {
      configureMaterial(model, material);
      ensureBodyForceAndRms(model);
      ensureCaseStudy(model, studyTag, nonlinear);
      model.study(studyTag).run();
    } catch (Exception e) {
      outError[0] = safeMsg(e);
      return false;
    }

    outMetrics[M_MAX_DISP] = evalWithFallback(
        model, "mxu_" + token, "MaxVolume", new String[]{"sqrt(u^2+v^2+w^2)"}
    );
    outMetrics[M_MAX_STRESS] = evalWithFallback(
        model, "mxs_" + token, "MaxVolume", new String[]{"solid.mises"}
    );
    outMetrics[M_MAX_STRAIN] = evalWithFallback(
        model, "mxe_" + token, "MaxVolume", new String[]{eqStrainExpr, "abs(solid.eel11)"}
    );
    if (Double.isFinite(outMetrics[M_MAX_STRESS]) && Double.isFinite(outMetrics[M_MAX_STRAIN])
        && Math.abs(outMetrics[M_MAX_STRAIN]) > 1e-16) {
      outMetrics[M_TANGENT] = outMetrics[M_MAX_STRESS] / outMetrics[M_MAX_STRAIN];
    }

    setMetricParam(
        model,
        token + "_max_disp",
        outMetrics[M_MAX_DISP],
        "[m]",
        "Max displacement magnitude"
    );
    setMetricParam(
        model,
        token + "_max_von_mises",
        outMetrics[M_MAX_STRESS],
        "[Pa]",
        "Max von Mises stress"
    );
    setMetricParam(
        model,
        token + "_max_strain",
        outMetrics[M_MAX_STRAIN],
        "",
        "Max equivalent strain"
    );
    setMetricParam(
        model,
        token + "_tangent_modulus",
        outMetrics[M_TANGENT],
        "[Pa]",
        "Tangent modulus = max_von_mises / max_strain"
    );

    return Double.isFinite(outMetrics[M_MAX_STRESS]);
  }

  private static String fmt(double value) {
    if (!Double.isFinite(value)) {
      return "NaN";
    }
    return String.format(Locale.US, "%.6e", value);
  }

  public static void main(String[] args) {
    Model model;
    try {
      model = ModelUtil.load("Model", TEMPLATE_MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load template: " + TEMPLATE_MPH, e);
    }

    int nEntity = ENTITY_NAMES.length;
    int nMat = MATERIALS.length;
    int nMode = MODES.length;
    boolean[][][] ok = new boolean[nEntity][nMat][nMode];
    String[][][] err = new String[nEntity][nMat][nMode];
    double[][][][] metrics = new double[nEntity][nMat][nMode][M_COUNT];

    for (int i = 0; i < nEntity; i++) {
      preflightOpen(model, ENTITY_NAMES[i], INPUT_BDFS[i]);
    }

    for (int i = 0; i < nEntity; i++) {
      String entity = ENTITY_NAMES[i];
      String solverBdf = SOLVER_BDFS[i];
      printKV(
          "ENTITY_START",
          new String[][]{
              {"entity", entity},
              {"solver_bdf", solverBdf}
          }
      );

      try {
        loadBdfForSolve(model, solverBdf);
      } catch (Exception e) {
        String loadErr = safeMsg(e);
        for (int m = 0; m < nMat; m++) {
          for (int k = 0; k < nMode; k++) {
            ok[i][m][k] = false;
            err[i][m][k] = "mesh load failed: " + loadErr;
          }
        }
        printKV(
            "ENTITY_DONE",
            new String[][]{
                {"entity", entity},
                {"ok", "false"},
                {"error", loadErr}
            }
        );
        continue;
      }

      for (int m = 0; m < nMat; m++) {
        String material = MATERIALS[m];
        for (int k = 0; k < nMode; k++) {
          String mode = MODES[k];
          double[] outMetrics = new double[M_COUNT];
          String[] outError = new String[1];

          printKV(
              "CASE_START",
              new String[][]{
                  {"entity", entity},
                  {"material", material},
                  {"mode", mode}
              }
          );

          ok[i][m][k] = runCase(model, entity, material, mode, outMetrics, outError);
          err[i][m][k] = outError[0] == null ? "" : outError[0];
          metrics[i][m][k] = outMetrics;

          printKV(
              "CASE_DONE",
              new String[][]{
                  {"entity", entity},
                  {"material", material},
                  {"mode", mode},
                  {"ok", Boolean.toString(ok[i][m][k])},
                  {"max_disp", fmt(outMetrics[M_MAX_DISP])},
                  {"max_von_mises", fmt(outMetrics[M_MAX_STRESS])},
                  {"max_strain", fmt(outMetrics[M_MAX_STRAIN])},
                  {"tangent_modulus", fmt(outMetrics[M_TANGENT])},
                  {"error", err[i][m][k]}
              }
          );

          try {
            model.save(OUTPUT_MPH);
            printKV(
                "CHECKPOINT_SAVE",
                new String[][]{
                    {"entity", entity},
                    {"material", material},
                    {"mode", mode},
                    {"ok", "true"}
                }
            );
          } catch (Exception e) {
            printKV(
                "CHECKPOINT_SAVE",
                new String[][]{
                    {"entity", entity},
                    {"material", material},
                    {"mode", mode},
                    {"ok", "false"},
                    {"error", safeMsg(e)}
                }
            );
          }
        }
      }

      printKV(
          "ENTITY_DONE",
          new String[][]{
              {"entity", entity},
              {"ok", "true"}
          }
      );
    }

    StringBuilder md = new StringBuilder();
    md.append("# Static Dynamics High Resolution Study Report\n\n");
    md.append("- Output model: `").append(OUTPUT_MPH).append("`\n");
    md.append("- Input meshes (unchanged):\n");
    for (String in : INPUT_BDFS) {
      md.append("  - `").append(in).append("`\n");
    }
    md.append("- Materials: `St. Venant-Kirchhoff`, `Mooney-Rivlin MR2`, `Mooney-Rivlin MR5`\n");
    md.append("- Modes: `linear`, `nonlinear`\n");
    md.append("- Metrics: max displacement, max von Mises, max equivalent strain, tangent modulus.\n\n");
    md.append("| Entity | Material | Mode | ok | max_disp (m) | max_von_mises (Pa) | max_strain | tangent_modulus (Pa) |\n");
    md.append("|---|---|---|---:|---:|---:|---:|---:|\n");
    for (int i = 0; i < nEntity; i++) {
      for (int m = 0; m < nMat; m++) {
        for (int k = 0; k < nMode; k++) {
          md.append("| ")
              .append(ENTITY_NAMES[i]).append(" | ")
              .append(MATERIALS[m]).append(" | ")
              .append(MODES[k]).append(" | ")
              .append(ok[i][m][k] ? "yes" : "no").append(" | ")
              .append(fmt(metrics[i][m][k][M_MAX_DISP])).append(" | ")
              .append(fmt(metrics[i][m][k][M_MAX_STRESS])).append(" | ")
              .append(fmt(metrics[i][m][k][M_MAX_STRAIN])).append(" | ")
              .append(fmt(metrics[i][m][k][M_TANGENT])).append(" |\n");
          if (!ok[i][m][k] && err[i][m][k] != null && !err[i][m][k].isEmpty()) {
            md.append("\n> ").append(ENTITY_NAMES[i]).append(" / ").append(MATERIALS[m]).append(" / ")
                .append(MODES[k]).append(" error: ").append(err[i][m][k]).append("\n\n");
          }
        }
      }
    }

    model.comments(md.toString());
    try {
      model.save(OUTPUT_MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model: " + OUTPUT_MPH, e);
    }
    try {
      Files.writeString(Path.of(OUTPUT_MD), md.toString());
    } catch (IOException e) {
      throw new RuntimeException("Failed to write report: " + OUTPUT_MD, e);
    }

    System.out.println("Saved: " + OUTPUT_MPH);
    System.out.println("Report: " + OUTPUT_MD);
  }
}
