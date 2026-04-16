import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Arrays;

public class ProbeStrict3BdfContactProps {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static String readProp(Object f, String p) {
    try {
      Method vt = f.getClass().getMethod("getValueType", String.class);
      String t = (String) vt.invoke(f, p);
      if ("String".equals(t)) {
        Method gs = f.getClass().getMethod("getString", String.class);
        String v = (String) gs.invoke(f, p);
        return t + ":" + String.valueOf(v);
      }
      if ("StringArray".equals(t)) {
        Method ga = f.getClass().getMethod("getStringArray", String.class);
        String[] v = (String[]) ga.invoke(f, p);
        return t + ":" + Arrays.toString(v);
      }
      if ("Boolean".equals(t)) {
        Method gb = f.getClass().getMethod("getBoolean", String.class);
        boolean v = (boolean) gb.invoke(f, p);
        return t + ":" + v;
      }
      if ("Double".equals(t)) {
        Method gd = f.getClass().getMethod("getDouble", String.class);
        double v = (double) gd.invoke(f, p);
        return t + ":" + v;
      }
      if ("DoubleArray".equals(t)) {
        Method gda = f.getClass().getMethod("getDoubleArray", String.class);
        double[] v = (double[]) gda.invoke(f, p);
        return t + ":" + Arrays.toString(v);
      }
      return t;
    } catch (Exception e) {
      return "ERR:" + e.getClass().getSimpleName() + ":" + e.getMessage();
    }
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    String[] tags = new String[] {"dcnt1", "dgcnt1"};
    for (String t : tags) {
      Object f = m.component("comp1").physics("solid").feature(t);
      String type = "";
      try {
        Method tm = f.getClass().getMethod("getType");
        type = (String) tm.invoke(f);
      } catch (Exception ignored) {}
      System.out.println("FEATURE|" + t + "|type=" + type);

      try {
        Method am = f.getClass().getMethod("isActive");
        boolean active = (boolean) am.invoke(f);
        System.out.println("ACTIVE|" + t + "|" + active);
      } catch (Exception e) {
        System.out.println("ACTIVE|" + t + "|err=" + e.getMessage());
      }

      try {
        Method sm = f.getClass().getMethod("selection");
        Object sel = sm.invoke(f);
        Method em = sel.getClass().getMethod("entities");
        int[] ent = (int[]) em.invoke(sel);
        System.out.println("SELECTION|" + t + "|count=" + (ent == null ? 0 : ent.length) + "|entities=" + Arrays.toString(ent));
      } catch (Exception e) {
        System.out.println("SELECTION|" + t + "|err=" + e.getMessage());
      }

      String[] props = new String[0];
      try {
        Method pm = f.getClass().getMethod("properties");
        props = (String[]) pm.invoke(f);
      } catch (Exception ignored) {}
      Arrays.sort(props);
      System.out.println("PROPS|" + t + "|" + Arrays.toString(props));
      for (String p : props) {
        String v = readProp(f, p);
        if (!"".equals(v) && !"null".equals(v)) {
          System.out.println("P|" + t + "|" + p + "|" + v);
        }
      }
    }
  }
}
