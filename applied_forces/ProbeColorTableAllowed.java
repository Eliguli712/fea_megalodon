import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeColorTableAllowed {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      try { m.result().remove("pg_ct_probe"); } catch (Exception e) {}
      m.result().create("pg_ct_probe", "PlotGroup3D");
      m.result("pg_ct_probe").create("surf1", "Surface");
      String[] vals = m.result("pg_ct_probe").feature("surf1").getAllowedPropertyValues("colortable");
      if (vals == null) {
        System.out.println("colortable allowed null");
      } else {
        System.out.println("colortable allowed count=" + vals.length);
        for (String v : vals) System.out.println(v);
      }
    } catch (Exception e) {
      System.out.println("probe failed: " + e.getMessage());
    }
  }
}
