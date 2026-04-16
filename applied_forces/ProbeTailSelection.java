import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeTailSelection {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").selection().remove("sel_tail_fix"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_tail_fix", "Box");
    m.component("comp1").selection("sel_tail_fix").set("entitydim", "2");
    m.component("comp1").selection("sel_tail_fix").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("zmin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("zmax", "4[m]");

    try {
      int[] b = m.component("comp1").selection("sel_tail_fix").entities(2);
      System.out.println("sel_tail_fix boundaries=" + (b == null ? -1 : b.length));
    } catch (Exception e) {
      System.out.println("read failed: " + e.getMessage());
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeTailSelection_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
