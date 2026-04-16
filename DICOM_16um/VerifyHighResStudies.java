import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class VerifyHighResStudies {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics_high_resolution.mph";

  private static final String[] ENTITIES = new String[]{
      "surface_mesh_smoothed",
      "tooth_surface_uncompressed",
      "tooth_surface_comsol_tet_vol"
  };
  private static final String[] MATERIALS = new String[]{"stvkirchhoff", "mr2", "mr5"};
  private static final String[] MODES = new String[]{"linear", "nonlinear"};

  private static String token(String entity, String mat, String mode) {
    return (entity + "_" + mat + "_" + mode).replaceAll("[^A-Za-z0-9_]", "_");
  }

  private static String getParamExpr(Model m, String name) {
    try {
      return m.param().get(name);
    } catch (Exception ignored) {
    }
    return "";
  }

  private static double evalParam(Model m, String name) {
    try {
      return m.param().evaluate(name);
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static String fmt(double v) {
    if (!Double.isFinite(v)) {
      return "NaN";
    }
    return String.format("%.6e", v);
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load: " + MODEL_PATH, e);
    }

    System.out.println("STUDIES|" + String.join(",", model.study().tags()));
    for (String study : new String[]{
        "std_stvkirchhoff_lin",
        "std_stvkirchhoff_nl",
        "std_mr2_lin",
        "std_mr2_nl",
        "std_mr5_lin",
        "std_mr5_nl"
    }) {
      boolean present = Arrays.asList(model.study().tags()).contains(study);
      System.out.println("STUDY_PRESENT|" + study + "|" + present);
    }

    int finiteCount = 0;
    int total = 0;
    for (String e : ENTITIES) {
      for (String m : MATERIALS) {
        for (String mode : MODES) {
          String t = token(e, m, mode);
          String pDisp = t + "_max_disp";
          String pVm = t + "_max_von_mises";
          String pStrain = t + "_max_strain";
          String pTan = t + "_tangent_modulus";

          double vDisp = evalParam(model, pDisp);
          double vVm = evalParam(model, pVm);
          double vStrain = evalParam(model, pStrain);
          double vTan = evalParam(model, pTan);
          boolean finite = Double.isFinite(vDisp) && Double.isFinite(vVm) && Double.isFinite(vStrain) && Double.isFinite(vTan);
          if (finite) {
            finiteCount++;
          }
          total++;
          System.out.println(
              "CASE_METRIC|entity=" + e
                  + "|material=" + m
                  + "|mode=" + mode
                  + "|finite=" + finite
                  + "|max_disp=" + fmt(vDisp)
                  + "|max_von_mises=" + fmt(vVm)
                  + "|max_strain=" + fmt(vStrain)
                  + "|tangent_modulus=" + fmt(vTan)
                  + "|expr_vm=" + getParamExpr(model, pVm)
          );
        }
      }
    }
    System.out.println("CASE_SUMMARY|finite_cases=" + finiteCount + "|total_cases=" + total);
  }
}
