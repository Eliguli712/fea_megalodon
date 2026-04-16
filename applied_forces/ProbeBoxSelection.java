import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeBoxSelection {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").selection().remove("sel_front_probe"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_front_probe", "Box");
    m.component("comp1").selection("sel_front_probe").set("entitydim", 2);
    m.component("comp1").selection("sel_front_probe").set("xmin", 0.0);
    m.component("comp1").selection("sel_front_probe").set("xmax", 100.0);
    m.component("comp1").selection("sel_front_probe").set("ymin", 0.0);
    m.component("comp1").selection("sel_front_probe").set("ymax", 100.0);
    m.component("comp1").selection("sel_front_probe").set("zmin", 21.4);
    m.component("comp1").selection("sel_front_probe").set("zmax", 100.0);

    int[] e = m.component("comp1").selection("sel_front_probe").entities(2);
    System.out.println("front boundaries=" + (e==null?0:e.length));
  }
}
