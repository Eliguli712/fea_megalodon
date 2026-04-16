import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Arrays;

public class ProbeStrict3BdfSolidFeatures {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static String safeStr(Object pf, String key) {
    try {
      Method m = pf.getClass().getMethod("getString", String.class);
      String v = (String) m.invoke(pf, key);
      return v == null ? "" : v;
    } catch (Exception ignored) {
      return "";
    }
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    String[] ft = m.component("comp1").physics("solid").feature().tags();
    System.out.println("SOLID_FEATURES|" + Arrays.toString(ft));
    for (String t : ft) {
      Object f = m.component("comp1").physics("solid").feature(t);
      String type = "";
      try {
        Method tm = f.getClass().getMethod("getType");
        type = (String) tm.invoke(f);
      } catch (Exception ignored) {}
      System.out.println("F|tag=" + t + "|type=" + type);
      try {
        Method sm = f.getClass().getMethod("selection");
        Object sel = sm.invoke(f);
        Method em = sel.getClass().getMethod("entities");
        int[] ent = (int[]) em.invoke(sel);
        System.out.println("SEL|tag=" + t + "|count=" + (ent == null ? 0 : ent.length));
      } catch (Exception e) {
        System.out.println("SEL|tag=" + t + "|err=" + e.getMessage());
      }
      String[] props = new String[0];
      try {
        Method pm = f.getClass().getMethod("properties");
        props = (String[]) pm.invoke(f);
      } catch (Exception ignored) {}
      for (String p : props) {
        if ("F".equals(p) || "FperVol".equals(p) || "p0".equals(p) || "pres".equals(p) || "geomnonlin".equals(p) || "MaterialModel".equals(p)) {
          String v = safeStr(f, p);
          if (!v.isEmpty()) {
            System.out.println("P|tag=" + t + "|k=" + p + "|v=" + v);
          } else {
            try {
              Method am = f.getClass().getMethod("getStringArray", String.class);
              String[] a = (String[]) am.invoke(f, p);
              if (a != null && a.length > 0) {
                System.out.println("A|tag=" + t + "|k=" + p + "|v=" + Arrays.toString(a));
              }
            } catch (Exception ignored2) {}
          }
        }
      }
    }
  }
}
