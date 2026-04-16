import com.comsol.model.*;
import com.comsol.model.util.*;

public class InspectMrPlots {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);
    String[] plots = {"pg_vms_std_mr2", "pg_vms_std_mr5"};
    for (String p : plots) {
      try {
        System.out.println("plot=" + p);
        try { System.out.println("  data=" + m.result(p).getString("data")); } catch (Exception ignored) {}
        try { System.out.println("  surf expr=" + m.result(p).feature("surf1").getString("expr")); } catch (Exception e) { System.out.println("  surf1 expr err="+e.getMessage()); }
        try { System.out.println("  surf descr=" + m.result(p).feature("surf1").getString("descr")); } catch (Exception ignored) {}
      } catch (Exception e) {
        System.out.println("plot " + p + " missing: " + e.getMessage());
      }
    }

    try {
      int dom = m.component("comp1").physics("solid").selection().entities(3).length;
      int bnd = m.component("comp1").physics("solid").selection().entities(2).length;
      System.out.println("solid selection dom=" + dom + " bnd=" + bnd);
    } catch (Exception e) {
      System.out.println("solid selection err=" + e.getMessage());
    }

    String[] dsets = m.result().dataset().tags();
    for (String ds: dsets) {
      try {
        String type = m.result().dataset(ds).getType();
        String sol = "";
        try { sol = m.result().dataset(ds).getString("solution"); } catch (Exception ignored) {}
        System.out.println("dataset " + ds + " type=" + type + " solution=" + sol);
      } catch (Exception ignored) {}
    }
  }
}
