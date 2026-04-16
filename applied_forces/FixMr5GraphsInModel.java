import com.comsol.model.*;
import com.comsol.model.util.*;

public class FixMr5GraphsInModel {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";

  private static final String FUNC_TAIL = "mr5ftail";
  private static final String FUNC_IMPACT = "mr5fimpact";
  private static final String FUNC_VM_MAX = "mr5fvmmax";
  private static final String FUNC_VM_AVG = "mr5fvmavg";
  private static final String FUNC_INST = "mr5finst";

  private static final String FNAME_TAIL = "mr5_trailing_force_fn";
  private static final String FNAME_IMPACT = "mr5_max_impact_fn";
  private static final String FNAME_VM_MAX = "mr5_max_mises_fn";
  private static final String FNAME_VM_AVG = "mr5_avg_mises_fn";
  private static final String FNAME_INST = "mr5_instant_impact_fn";

  private static final String PG_TAIL = "pg_mr5_trailing_force_img";
  private static final String PG_IMPACT = "pg_mr5_max_impact_img";
  private static final String PG_VM = "pg_mr5_von_mises_img";
  private static final String PG_INST = "pg_mr5_instant_impact_img";
  private static final String PG_VM_TMP = "pg_mr5_von_mises_avg_seed";

  private static final String[][] TAIL = new String[][]{
    {"500.0", "9366.115806784263"},
    {"1000.0", "9440.063677720913"},
    {"1500.0", "9514.15978598043"},
    {"2000.0", "9588.395657788456"},
    {"2500.0", "9662.763795153596"},
    {"3000.0", "9737.257499874824"},
    {"3500.0", "9811.870716507927"},
    {"4000.0", "9886.597929312647"}
  };
  private static final String[][] IMPACT = new String[][]{
    {"500.0", "78357.95704241878"},
    {"1000.0", "79127.77209531023"},
    {"1500.0", "79901.39107785346"},
    {"2000.0", "80678.55078111622"},
    {"2500.0", "81459.02383281045"},
    {"3000.0", "82242.62273088755"},
    {"3500.0", "83029.17096993951"},
    {"4000.0", "83818.51488400821"}
  };
  private static final String[][] VM_MAX = new String[][]{
    {"500.0", "109282.27583604983"},
    {"1000.0", "112087.31684560276"},
    {"1500.0", "114892.9404631258"},
    {"2000.0", "117699.09564202577"},
    {"2500.0", "120505.73608695202"},
    {"3000.0", "123312.819714992"},
    {"3500.0", "126120.30818850402"},
    {"4000.0", "128928.16650881164"}
  };
  private static final String[][] VM_AVG = new String[][]{
    {"500.0", "4116.991406821023"},
    {"1000.0", "4196.738417163042"},
    {"1500.0", "4276.7558974128"},
    {"2000.0", "4357.018593492541"},
    {"2500.0", "4437.504894913516"},
    {"3000.0", "4518.196158982152"},
    {"3500.0", "4599.076146118097"},
    {"4000.0", "4680.130578972834"}
  };
  private static final String[][] INST = new String[][]{
    {"500.0", "107384.09760479114"},
    {"1000.0", "110131.18297238447"},
    {"1500.0", "112878.32001805623"},
    {"2000.0", "115625.49787510022"},
    {"2500.0", "118372.70668729082"},
    {"3000.0", "121119.9374939167"},
    {"3500.0", "123867.1821307975"},
    {"4000.0", "126614.4331437453"}
  };

  private static void log(String s) { System.out.println(s); }
  private static void safeRemoveResult(Model m, String tag) { try { m.result().remove(tag); } catch (Exception ignored) {} }
  private static void safeRemoveFunc(Model m, String tag) { try { m.component("comp1").func().remove(tag); } catch (Exception ignored) {} }
  private static void safeSet(PropFeature f, String key, String val) { try { f.set(key, val); } catch (Exception ignored) {} }
  private static void safeSet(PropFeature f, String key, String[] val) { try { f.set(key, val); } catch (Exception ignored) {} }

  private static FunctionFeature createInterpolation(Model m, String tag, String label, String funcName, String funUnit, String[][] table) {
    safeRemoveFunc(m, tag);
    FunctionFeature f = m.component("comp1").func().create(tag, "Interpolation");
    f.label(label);
    f.set("source", "table");
    f.set("funcname", funcName);
    f.set("argunit", new String[]{"N"});
    f.set("fununit", new String[]{funUnit});
    f.set("interp", "linear");
    f.set("extrap", "const");
    f.set("table", table);
    return f;
  }

  private static void stylePlotGroup(ResultFeature pg, String title, String yLabel) {
    pg.label(title);
    safeSet(pg, "titletype", "custom");
    safeSet(pg, "title", title);
    safeSet(pg, "xlabel", "Front-end stress (N)");
    safeSet(pg, "ylabel", yLabel);
    safeSet(pg, "showlegends", "on");
  }

  private static void styleCurve(ResultFeature curve, String curveLabel, String expr, String unit, String minX, String maxX) {
    curve.label(curveLabel);
    safeSet(curve, "expr", expr);
    safeSet(curve, "unit", unit);
    safeSet(curve, "descractive", "on");
    safeSet(curve, "descr", curveLabel);
    safeSet(curve, "xdataexpr", "t");
    safeSet(curve, "xdataunit", "N");
    safeSet(curve, "xdatadescractive", "on");
    safeSet(curve, "xdatadescr", "Front-end stress");
    safeSet(curve, "lowerbound", minX);
    safeSet(curve, "upperbound", maxX);
    safeSet(curve, "display", "linepoints");
    safeSet(curve, "legend", "on");
    safeSet(curve, "legends", new String[]{curveLabel});
  }

  private static void buildSinglePlot(Model m, FunctionFeature f, String plotTag, String title, String curveLabel, String expr, String unit, String minX, String maxX) {
    safeRemoveResult(m, plotTag);
    ResultFeature pg = f.createPlot(plotTag);
    stylePlotGroup(pg, title, curveLabel + " (" + unit + ")");
    styleCurve(pg.feature("plot1"), curveLabel, expr, unit, minX, maxX);
    pg.run();
    log(plotTag + " warning=" + pg.feature("plot1").hasWarning());
  }

  private static void buildDualVonMisesPlot(Model m, FunctionFeature fMax, FunctionFeature fAvg, String minX, String maxX) {
    safeRemoveResult(m, PG_VM);
    safeRemoveResult(m, PG_VM_TMP);

    ResultFeature pg = fMax.createPlot(PG_VM);
    ResultFeature seed = fAvg.createPlot(PG_VM_TMP);
    seed.run();
    m.result().remove(PG_VM_TMP);

    stylePlotGroup(pg, "MR5 Max and Avg von Mises vs Front-End Stress", "von Mises (Pa)");
    styleCurve(pg.feature("plot1"), "Max von Mises", "comp1." + FNAME_VM_MAX + "(t)", "Pa", minX, maxX);

    ResultFeature p2 = pg.create("plot2", "Function");
    p2.set("data", FUNC_VM_AVG + "_ds1");
    styleCurve(p2, "Avg von Mises", "comp1." + FNAME_VM_AVG + "(t)", "Pa", minX, maxX);

    pg.run();
    log(PG_VM + " plot1 warning=" + pg.feature("plot1").hasWarning());
    log(PG_VM + " plot2 warning=" + pg.feature("plot2").hasWarning());
  }

  public static void main(String[] args) throws Exception {
    String minX = TAIL[0][0];
    String maxX = TAIL[TAIL.length - 1][0];
    Model m = ModelUtil.load("Model", MPH);

    FunctionFeature fTail = createInterpolation(m, FUNC_TAIL, "MR5 Trailing Edge Force", FNAME_TAIL, "N", TAIL);
    FunctionFeature fImpact = createInterpolation(m, FUNC_IMPACT, "MR5 Max Impact", FNAME_IMPACT, "N*m", IMPACT);
    FunctionFeature fVmMax = createInterpolation(m, FUNC_VM_MAX, "MR5 Max von Mises", FNAME_VM_MAX, "Pa", VM_MAX);
    FunctionFeature fVmAvg = createInterpolation(m, FUNC_VM_AVG, "MR5 Avg von Mises", FNAME_VM_AVG, "Pa", VM_AVG);
    FunctionFeature fInst = createInterpolation(m, FUNC_INST, "MR5 Instantaneous Impact", FNAME_INST, "W/m^2", INST);

    buildSinglePlot(m, fTail, PG_TAIL,
      "MR5 Trailing Edge Force vs Front-End Stress",
      "Max trailing edge force",
      "comp1." + FNAME_TAIL + "(t)", "N", minX, maxX);

    buildSinglePlot(m, fImpact, PG_IMPACT,
      "MR5 Max Impact vs Front-End Stress",
      "Max impact",
      "comp1." + FNAME_IMPACT + "(t)", "N*m", minX, maxX);

    buildDualVonMisesPlot(m, fVmMax, fVmAvg, minX, maxX);

    buildSinglePlot(m, fInst, PG_INST,
      "MR5 Instantaneous Impact vs Front-End Stress",
      "Instantaneous impact",
      "comp1." + FNAME_INST + "(t)", "W/m^2", minX, maxX);

    m.save(MPH);
    log("Saved " + MPH);
  }
}
