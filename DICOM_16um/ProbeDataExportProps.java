import com.comsol.model.*;
import com.comsol.model.util.*;

import java.lang.reflect.Method;
import java.util.Arrays;

public class ProbeDataExportProps {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  private static void invokeSet(Object ex, String key, String[] value) throws Exception {
    Method m = ex.getClass().getMethod("set", String.class, String[].class);
    m.invoke(ex, key, value);
  }

  private static void invokeSet(Object ex, String key, String value) throws Exception {
    Method m = ex.getClass().getMethod("set", String.class, String.class);
    m.invoke(ex, key, value);
  }

  private static void invokeSet(Object ex, String key, boolean value) throws Exception {
    Method m = ex.getClass().getMethod("set", String.class, boolean.class);
    m.invoke(ex, key, value);
  }

  private static String[] invokeProperties(Object ex) throws Exception {
    Method m = ex.getClass().getMethod("properties");
    return (String[]) m.invoke(ex);
  }

  private static String[] invokeAllowed(Object ex, String key) throws Exception {
    Method m = ex.getClass().getMethod("getAllowedPropertyValues", String.class);
    return (String[]) m.invoke(ex, key);
  }

  private static void invokeRun(Object ex) throws Exception {
    Method m = ex.getClass().getMethod("run");
    m.invoke(ex);
  }

  public static void main(String[] args) throws Exception {
    Model model = ModelUtil.load("Model", MODEL_PATH);
    try {
      model.result().export().remove("tmpd");
    } catch (Exception ignored) {
    }
    model.result().export().create("tmpd", "Data");
    Object ex = model.result().export("tmpd");

    try {
      String[] props = invokeProperties(ex);
      System.out.println("DATA_EXPORT_PROPERTIES|" + String.join(",", props));
    } catch (Exception e) {
      System.out.println("DATA_EXPORT_PROPERTIES|error|" + e.getMessage());
    }

    for (String key : new String[]{
        "data", "expr", "descr", "filename", "location", "gridx3", "gridy3", "gridz3", "unit", "field",
        "outerinput", "outersolnum", "innerinput", "solnum", "t"
    }) {
      try {
        String[] allowed = invokeAllowed(ex, key);
        System.out.println("ALLOWED|" + key + "|" + Arrays.toString(allowed));
      } catch (Exception e) {
        System.out.println("ALLOWED|" + key + "|error|" + e.getMessage());
      }
    }

    try {
      invokeSet(ex, "data", "dset4");
      invokeSet(ex, "expr", new String[]{"solid.mises"});
      invokeSet(ex, "filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/probe_data_export.txt");
      invokeSet(ex, "location", "fromdataset");
      invokeSet(ex, "header", true);
      invokeRun(ex);
      System.out.println("DATA_EXPORT_RUN|ok");
    } catch (Exception e) {
      System.out.println("DATA_EXPORT_RUN|error|" + e.getMessage());
    }
  }
}
