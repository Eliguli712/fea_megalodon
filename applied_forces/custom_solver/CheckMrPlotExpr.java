import com.comsol.model.*;
import com.comsol.model.util.*;

public class CheckMrPlotExpr {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);
    String[] p = {"pg_vms_std_mr2", "pg_vms_std_mr5"};
    for (String t : p) {
      System.out.println(t + " expr=" + m.result(t).feature("surf1").getString("expr"));
      System.out.println(t + " descr=" + m.result(t).feature("surf1").getString("descr"));
    }
  }
}
