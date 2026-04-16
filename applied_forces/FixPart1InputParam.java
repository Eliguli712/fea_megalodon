import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class FixPart1InputParam {
  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/FixPart1InputParam_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      m.geom("part1").inputParam().set("solid", "0");
      m.geom("part1").inputParam().set("endsolid", "0");
      m.geom("part1").inputParam().set("facet", "0");
      m.geom("part1").inputParam().set("outer", "0");
      m.geom("part1").inputParam().set("vertex", "0");
      System.out.println("inputParam patched");
    } catch (Exception e) {
      System.out.println("inputParam patch failed: " + e.getMessage());
    }

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
