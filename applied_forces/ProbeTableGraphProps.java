import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeTableGraphProps {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.result().remove("pg_probe_1d"); } catch (Exception e) {}
    m.result().create("pg_probe_1d", "PlotGroup1D");
    System.out.println("PG1D ok");
    String[] ops = new String[]{"TableGraph","LineGraph","PointGraph","Global","TableSurface","Plot"};
    for (String op: ops) {
      try {
        try { m.result("pg_probe_1d").feature().remove("f1"); } catch (Exception ex) {}
        m.result("pg_probe_1d").create("f1", op);
        System.out.println("OK op=" + op + " type=" + m.result("pg_probe_1d").feature("f1").getType());
        String[] props = m.result("pg_probe_1d").feature("f1").properties();
        System.out.println("PROPS " + op);
        for (String p : props) System.out.println("  " + p + " type=" + m.result("pg_probe_1d").feature("f1").getValueType(p));
        for (String key : new String[]{"table","xdata","xcol","ycol","legend","legendmethod","type","expr","descr","unit"}) {
          try {
            System.out.println("ALLOWED " + op + " " + key + " -> " + java.util.Arrays.toString(m.result("pg_probe_1d").feature("f1").getAllowedPropertyValues(key)));
          } catch (Exception ee) {
            System.out.println("ALLOWED " + op + " " + key + " -> <err> " + ee.getMessage());
          }
        }
      } catch (Exception e) {
        System.out.println("BAD op=" + op + " msg=" + e.getMessage());
      }
    }
  }
}
