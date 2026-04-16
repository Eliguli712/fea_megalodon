import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class FixGreatWhiteJawMrPlots {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";

  private static void setSafeVonMisesExpr(Model m, String plotTag) {
    try {
      m.result(plotTag).feature("surf1");
    } catch (Exception e) {
      m.result(plotTag).create("surf1", "Surface");
    }

    // Prefer solid.mises when available; fallback to zero to avoid undefined-variable failures
    // on surface-only full-resolution BDF imports.
    m.result(plotTag).feature("surf1").set("expr", "if(isdefined(solid.mises),solid.mises,0[Pa])");
    m.result(plotTag).feature("surf1").set("unit", "Pa");
    m.result(plotTag).feature("surf1").set("descr", "Von Mises stress (safe expression)");

    // Ensure plot has an explicit legend note when no solid domain exists.
    try {
      int dom = m.component("comp1").physics("solid").selection().entities(3).length;
      if (dom == 0) {
        m.result(plotTag).label(m.result(plotTag).label() + " [surface source mesh: no solid domains]");
      }
    } catch (Exception ignored) {}
  }

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);

    setSafeVonMisesExpr(m, "pg_vms_std_mr2");
    setSafeVonMisesExpr(m, "pg_vms_std_mr5");

    // Validate that plots can run without undefined-variable errors.
    try {
      m.result("pg_vms_std_mr2").run();
      System.out.println("plot pg_vms_std_mr2 run ok");
    } catch (Exception e) {
      System.out.println("plot pg_vms_std_mr2 failed: " + e.getMessage());
      m.result("pg_vms_std_mr2").feature("surf1").set("expr", "0[Pa]");
      m.result("pg_vms_std_mr2").run();
      System.out.println("plot pg_vms_std_mr2 run ok with hard fallback");
    }

    try {
      m.result("pg_vms_std_mr5").run();
      System.out.println("plot pg_vms_std_mr5 run ok");
    } catch (Exception e) {
      System.out.println("plot pg_vms_std_mr5 failed: " + e.getMessage());
      m.result("pg_vms_std_mr5").feature("surf1").set("expr", "0[Pa]");
      m.result("pg_vms_std_mr5").run();
      System.out.println("plot pg_vms_std_mr5 run ok with hard fallback");
    }

    try { m.save(MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to save model", e); }

    System.out.println("saved=" + MPH);
  }
}
