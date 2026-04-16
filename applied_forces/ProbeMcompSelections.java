import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeMcompSelections {
  private static void log(String s) { System.out.println(s); }

  public static Model run() {
    String mph = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model m;
    try { m = ModelUtil.load("Model", mph); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.selection("mpart1_imp1___tracked_surface_stl"); log("selection boundary exists"); }
    catch (Exception e) { log("selection boundary missing: " + e.getMessage()); }

    try { m.selection("mpart1_imp1___tracked_surface_stl_1"); log("selection domain exists"); }
    catch (Exception e) { log("selection domain missing: " + e.getMessage()); }

    try {
      m.component("comp1").physics("solid").selection().named("mpart1_imp1___tracked_surface_stl_1");
      log("solid domain selection set to mpart1 domain selection");
    } catch (Exception e) {
      log("solid domain named selection failed: " + e.getMessage());
    }

    try {
      m.component("comp1").physics("solid").feature("bndl1").selection().named("mpart1_imp1___tracked_surface_stl");
      log("bndl1 boundary selection set to mpart1 boundary selection");
    } catch (Exception e) {
      log("bndl1 named selection failed: " + e.getMessage());
    }

    try {
      m.component("comp1").physics("solid").feature("bndl1").set("forceType", "ForceArea");
      m.component("comp1").physics("solid").feature("bndl1").set("force_src", "userdef");
      m.component("comp1").physics("solid").feature("bndl1").set("force", new String[]{"0","0","thrust_load"});
      m.component("comp1").physics("solid").feature("bndl1").active(true);
      try { m.component("comp1").physics("solid").feature("bndl_pr").active(false); } catch (Exception ex) {}
      log("bndl1 configured");
    } catch (Exception e) {
      log("bndl1 configure failed: " + e.getMessage());
    }

    try {
      m.component("comp1").physics("solid").create("fix_test", "Fixed", 2);
      log("fixed feature create succeeded");
      try {
        m.component("comp1").physics("solid").feature("fix_test").selection().named("mpart1_imp1___tracked_surface_stl");
        log("fix_test named selection set");
      } catch (Exception e) {
        log("fix_test named selection failed: " + e.getMessage());
      }
    } catch (Exception e) {
      log("fixed feature create failed: " + e.getMessage());
    }

    try {
      m.component("comp1").physics("solid").create("rms_test", "RigidMotionSuppression", 3);
      log("rigid motion suppression create succeeded");
      try {
        m.component("comp1").physics("solid").feature("rms_test").selection().named("mpart1_imp1___tracked_surface_stl_1");
        log("rms_test named selection set");
      } catch (Exception e) {
        log("rms_test named selection failed: " + e.getMessage());
      }
    } catch (Exception e) {
      log("rigid motion suppression create failed: " + e.getMessage());
    }

    try {
      m.study("std1").run();
      log("std1 run completed");
    } catch (Exception e) {
      log("std1 run failed: " + e.getMessage());
    }

    try {
      m.result().numerical().create("max_probe", "MaxVolume");
      m.result().numerical("max_probe").set("expr", new String[]{"solid.mises"});
      m.result().numerical("max_probe").setResult();
      double[][] val = m.result().numerical("max_probe").getReal();
      if (val != null && val.length > 0 && val[0].length > 0) {
        log("max solid.mises = " + val[0][0]);
      } else {
        log("max solid.mises unavailable");
      }
    } catch (Exception e) {
      log("mises eval failed: " + e.getMessage());
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics_probe_mcomp.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    return m;
  }

  public static void main(String[] args) { run(); }
}
