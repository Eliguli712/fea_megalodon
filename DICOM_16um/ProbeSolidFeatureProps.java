import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Arrays;

public class ProbeSolidFeatureProps {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  private static String[] featureProps(Object feature) {
    try {
      Method m = feature.getClass().getMethod("properties");
      return (String[]) m.invoke(feature);
    } catch (Exception ignored) {
    }
    return new String[0];
  }

  private static String[] allowed(Object feature, String key) {
    try {
      Method m = feature.getClass().getMethod("getAllowedPropertyValues", String.class);
      return (String[]) m.invoke(feature, key);
    } catch (Exception ignored) {
    }
    return null;
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load: " + MODEL_PATH, e);
    }

    for (String tag : new String[]{"lemm1", "hmm_mr2", "hmm_mr5"}) {
      try {
        Object f = model.component("comp1").physics("solid").feature(tag);
        String[] props = featureProps(f);
        System.out.println("FEATURE|" + tag + "|props_count=" + props.length);
        System.out.println("FEATURE|" + tag + "|props=" + String.join(",", props));
        for (String key : new String[]{
            "MaterialModel",
            "Compressibility_MooneyRivlin",
            "YoungsModulus",
            "poissonsratio",
            "E_mat",
            "nu_mat",
            "C10_mat",
            "C01_mat",
            "C20_mat",
            "C11_mat",
            "C02_mat",
            "kappa"
        }) {
          String[] av = allowed(f, key);
          System.out.println("ALLOWED|" + tag + "|" + key + "|" + (av == null ? "null" : Arrays.toString(av)));
        }
      } catch (Exception e) {
        System.out.println("FEATURE|" + tag + "|error=" + e.getMessage());
      }
    }

    try {
      Object stat = model.study("std_mr5").feature("stat");
      String[] props = featureProps(stat);
      System.out.println("STUDY|std_mr5/stat|props=" + String.join(",", props));
      for (String key : new String[]{"geometricNonlinearity", "geomnonlin", "shapeorder"}) {
        String[] av = allowed(stat, key);
        System.out.println("ALLOWED|study|" + key + "|" + (av == null ? "null" : Arrays.toString(av)));
      }
    } catch (Exception e) {
      System.out.println("STUDY|error=" + e.getMessage());
    }
  }
}
