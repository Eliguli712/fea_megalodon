import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeGraphOps {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    String[] rootOps = new String[]{"PlotGroup1D","PlotGroup2D","PlotGroup3D","TableGraph","Image","Data","Table"};
    for (String op: rootOps) {
      try {
        try { m.result().remove("rt"); } catch (Exception ex) {}
        m.result().create("rt", op);
        System.out.println("ROOT_OK op=" + op + " type=" + m.result("rt").getType());
      } catch (Exception e) {
        System.out.println("ROOT_BAD op=" + op + " msg=" + e.getMessage());
      }
    }

    String[] pg1Ops = new String[]{"TableGraph","LineGraph","Global","PointGraph","Image","TableSurface"};
    try { m.result().remove("pg1"); } catch (Exception e) {}
    m.result().create("pg1", "PlotGroup1D");
    for (String op: pg1Ops) {
      try {
        try { m.result("pg1").feature().remove("f1"); } catch (Exception ex) {}
        m.result("pg1").create("f1", op);
        System.out.println("PG1_OK op=" + op + " type=" + m.result("pg1").feature("f1").getType());
      } catch (Exception e) {
        System.out.println("PG1_BAD op=" + op + " msg=" + e.getMessage());
      }
    }

    String[] pg2Ops = new String[]{"TableGraph","LineGraph","Surface","Image","Contour","HeightExpression","TableSurface"};
    try { m.result().remove("pg2"); } catch (Exception e) {}
    m.result().create("pg2", "PlotGroup2D");
    for (String op: pg2Ops) {
      try {
        try { m.result("pg2").feature().remove("f1"); } catch (Exception ex) {}
        m.result("pg2").create("f1", op);
        System.out.println("PG2_OK op=" + op + " type=" + m.result("pg2").feature("f1").getType());
      } catch (Exception e) {
        System.out.println("PG2_BAD op=" + op + " msg=" + e.getMessage());
      }
    }
  }
}
