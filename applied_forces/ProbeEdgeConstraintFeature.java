import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeEdgeConstraintFeature {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      try { m.component("comp1").physics("solid").feature().remove("fixe_test"); } catch (Exception e) {}
      m.component("comp1").physics("solid").create("fixe_test", "Fixed", 1);
      m.component("comp1").physics("solid").feature("fixe_test").selection().geom("geom1",1);
      m.component("comp1").physics("solid").feature("fixe_test").selection().all();
      System.out.println("Fixed dim1 created OK");
    } catch (Exception e) {
      System.out.println("Fixed dim1 failed: " + e.getMessage());
    }

    try {
      try { m.component("comp1").physics("solid").feature().remove("disp_test"); } catch (Exception e) {}
      m.component("comp1").physics("solid").create("disp_test", "PrescribedDisplacement", 1);
      m.component("comp1").physics("solid").feature("disp_test").selection().geom("geom1",1);
      m.component("comp1").physics("solid").feature("disp_test").selection().all();
      m.component("comp1").physics("solid").feature("disp_test").set("U0", new String[]{"0","0","0"});
      System.out.println("PrescribedDisplacement dim1 created OK");
    } catch (Exception e) {
      System.out.println("PrescribedDisplacement dim1 failed: " + e.getMessage());
    }

    try {
      m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeEdgeConstraintFeature_Model.mph");
    } catch (IOException e) { throw new RuntimeException(e); }
  }
}
