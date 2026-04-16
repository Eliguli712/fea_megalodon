import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeStrict3BdfCountsNoRun {
  private static int countAll(Model m, int dim, String tag) {
    try {
      try { m.component("comp1").selection().remove(tag); } catch (Exception ignored) {}
      m.component("comp1").selection().create(tag, "Explicit");
      m.component("comp1").selection(tag).geom("geom1", dim);
      m.component("comp1").selection(tag).all();
      int[] e = m.component("comp1").selection(tag).entities();
      return e == null ? 0 : e.length;
    } catch (Exception ignored) {
      return -1;
    }
  }

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    int b = countAll(m, 2, "tmp_bnd_probe");
    int d = countAll(m, 3, "tmp_dom_probe");
    System.out.println("COUNTS|bnd=" + b + "|dom=" + d + "|nonzero=" + (b > 0 && d > 0));
  }
}
