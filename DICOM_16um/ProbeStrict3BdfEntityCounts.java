import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class ProbeStrict3BdfEntityCounts {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

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
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    int b0 = countAll(m, 2, "tmp_b0");
    int d0 = countAll(m, 3, "tmp_d0");
    System.out.println("COUNTS_BEFORE|bnd=" + b0 + "|dom=" + d0);

    try {
      m.mesh("mpart1").run("imp1");
      m.mesh("mpart1").run();
      m.component("comp1").mesh("mesh1").run("impmsh");
      m.component("comp1").mesh("mesh1").run();
      m.component("comp1").mesh("mesh2").run();
      System.out.println("MESH_RUN|ok=true");
    } catch (Exception e) {
      System.out.println("MESH_RUN|ok=false|err=" + e.getMessage());
    }

    int b1 = countAll(m, 2, "tmp_b1");
    int d1 = countAll(m, 3, "tmp_d1");
    System.out.println("COUNTS_AFTER|bnd=" + b1 + "|dom=" + d1);
  }
}
