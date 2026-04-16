import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeMergeFunctionCurves {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String OUT = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeMergeFunctionCurves.out.mph";

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
    try { m.result().remove("pg_probe_merge1"); } catch (Exception ignored) {}
    try { m.result().remove("pg_probe_merge2"); } catch (Exception ignored) {}

    FunctionFeature f1 = m.component("comp1").func().create("fmaxprobe", "Interpolation");
    setTable(f1, "fmaxprobev", "Pa", new String[][]{{"500","100"},{"1000","200"},{"1500","300"}});
    FunctionFeature f2 = m.component("comp1").func().create("favgprobe", "Interpolation");
    setTable(f2, "favgprobev", "Pa", new String[][]{{"500","10"},{"1000","20"},{"1500","30"}});

    ResultFeature pg1 = f1.createPlot("pg_probe_merge1");
    ResultFeature pg2 = f2.createPlot("pg_probe_merge2");
    pg2.run();
    m.result().remove("pg_probe_merge2");

    ResultFeature p2 = pg1.create("plot2", "Function");
    p2.label("Avg curve");
    p2.set("data", "favgprobe_ds1");
    p2.set("expr", "comp1.favgprobev(t)");
    p2.set("unit", "Pa");
    p2.set("descractive", "on");
    p2.set("descr", "Avg curve");
    p2.set("xdataexpr", "t");
    p2.set("xdataunit", "N");
    p2.set("xdatadescractive", "on");
    p2.set("xdatadescr", "Front stress");
    p2.set("lowerbound", "500");
    p2.set("upperbound", "1500");
    p2.set("display", "linepoints");
    p2.set("legend", "on");
    p2.set("legends", new String[]{"Avg curve"});

    pg1.run();
    System.out.println("plot1 warning=" + pg1.feature("plot1").hasWarning());
    System.out.println("plot2 warning=" + pg1.feature("plot2").hasWarning());
    m.save(OUT);
    System.out.println("saved");
  }
}
