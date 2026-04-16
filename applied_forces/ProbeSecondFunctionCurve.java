import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeSecondFunctionCurve {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String OUT = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeSecondFunctionCurve.out.mph";

  private static void setTable(FunctionFeature f, String funcname, String fununit, String[][] rows) {
    f.set("source", "table");
    f.set("funcname", funcname);
    f.set("argunit", new String[]{"N"});
    f.set("fununit", new String[]{fununit});
    f.set("table", rows);
  }

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);
    try { m.component("comp1").func().remove("fmaxprobe"); } catch (Exception ignored) {}
    try { m.component("comp1").func().remove("favgprobe"); } catch (Exception ignored) {}
    try { m.result().remove("pg_probe_twofun"); } catch (Exception ignored) {}
    try { m.result().remove("pg_probe_twofun_2"); } catch (Exception ignored) {}

    FunctionFeature f1 = m.component("comp1").func().create("fmaxprobe", "Interpolation");
    setTable(f1, "fmaxprobev", "Pa", new String[][]{{"500","100"},{"1000","200"},{"1500","300"}});
    FunctionFeature f2 = m.component("comp1").func().create("favgprobe", "Interpolation");
    setTable(f2, "favgprobev", "Pa", new String[][]{{"500","10"},{"1000","20"},{"1500","30"}});

    ResultFeature pg1 = f1.createPlot("pg_probe_twofun");
    System.out.println("pg1 tag=" + pg1.tag() + " type=" + pg1.getType());
    ResultFeature pg2 = f2.createPlot("pg_probe_twofun_2");
    System.out.println("pg2 tag=" + pg2.tag() + " type=" + pg2.getType());
    pg1.run();
    pg2.run();
    System.out.println("pg1 plot1 warning=" + pg1.feature("plot1").hasWarning());
    System.out.println("pg2 plot1 warning=" + pg2.feature("plot1").hasWarning());
    m.save(OUT);
    System.out.println("saved");
  }
}
