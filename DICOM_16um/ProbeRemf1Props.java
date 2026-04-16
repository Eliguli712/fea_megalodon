import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class ProbeRemf1Props {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static String safeGet(PropFeature pf, String key) {
    try {
      String v = pf.getString(key);
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
    PropFeature f = m.mesh("mpart1").feature("remf1");
    String type = "";
    try { type = f.getType(); } catch (Exception ignored) {}
    System.out.println("TYPE|" + type);
    String[] props = new String[0];
    try { props = f.properties(); } catch (Exception ignored) {}
    System.out.println("PROPS|" + Arrays.toString(props));
    for (String p : props) {
      String s = safeGet(f, p);
      if (!s.isEmpty()) {
        System.out.println("P|" + p + "|" + s);
      } else {
        try {
          String[] a = f.getStringArray(p);
          if (a != null && a.length > 0) {
            System.out.println("A|" + p + "|" + Arrays.toString(a));
          }
        } catch (Exception ignored) {}
      }
    }
    try {
      int[] sel = f.selection().entities();
      System.out.println("SEL_COUNT|" + (sel == null ? 0 : sel.length));
      if (sel != null && sel.length > 0) {
        System.out.println("SEL_FIRST|" + sel[0]);
      }
    } catch (Exception e) {
      System.out.println("SEL_ERR|" + e.getMessage());
    }
  }
}
