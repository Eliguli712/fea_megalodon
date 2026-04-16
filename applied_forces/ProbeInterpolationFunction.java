import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;

public class ProbeInterpolationFunction {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String OUTMPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeInterpolationFunction.out.mph";

  private static void p(String s) { System.out.println(s); }
  private static void trySet(PropFeature f, String key, Object value) {
    try {
      if (value instanceof String) f.set(key, (String) value);
      else if (value instanceof String[]) f.set(key, (String[]) value);
      else if (value instanceof String[][]) f.set(key, (String[][]) value);
      else if (value instanceof Integer) f.set(key, ((Integer) value).intValue());
      else if (value instanceof Boolean) f.set(key, ((Boolean) value).booleanValue());
      else if (value instanceof Double) f.set(key, ((Double) value).doubleValue());
      p("SET OK " + key + " = " + String.valueOf(value));
    } catch (Exception e) {
      p("SET BAD " + key + " = " + String.valueOf(value) + " :: " + e.getMessage());
    }
  }

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);
    try { m.component("comp1").func().remove("fprobe"); } catch (Exception ignored) {}
    try { m.result().remove("pg_probe_fun"); } catch (Exception ignored) {}

    FunctionFeature f = m.component("comp1").func().create("fprobe", "Interpolation");
    f.label("Probe Interpolation Function");
    p("TYPE=" + f.getType());
    String[] props = f.properties();
    Arrays.sort(props);
    p("PROP COUNT=" + props.length);
    for (String key : props) {
      String vt = "";
      String[] allowed = null;
      try { vt = f.getValueType(key); } catch (Exception e) { vt = "<err:" + e.getMessage() + ">"; }
      try { allowed = f.getAllowedPropertyValues(key); } catch (Exception ignored) {}
      if (key.contains("table") || key.contains("source") || key.contains("file") || key.contains("arg") || key.contains("fun") || key.contains("unit") || key.contains("interp") || key.contains("narg") || key.contains("extrap")) {
        p("PROP " + key + " type=" + vt + " allowed=" + (allowed == null ? "null" : Arrays.toString(allowed)));
      }
    }
    for (String key : new String[]{"source","table","filename","funcname","argunit","fununit","interp","extrap","nargs","arg","defvars"}) {
      try {
        p("KEY " + key + " type=" + f.getValueType(key) + " allowed=" + Arrays.toString(f.getAllowedPropertyValues(key)));
      } catch (Exception e) {
        p("KEY " + key + " <err> " + e.getMessage());
      }
    }

    trySet(f, "source", "table");
    trySet(f, "funcname", "mr5trailprobe");
    trySet(f, "argunit", new String[]{"N"});
    trySet(f, "fununit", "N");
    trySet(f, "table", new String[][]{
      {"500", "9366.115806784263"},
      {"1000", "9440.063677720913"},
      {"1500", "9514.15978598043"}
    });
    try { f.run(); p("RUN OK"); } catch (Exception e) { p("RUN BAD " + e.getMessage()); }

    try {
      ResultFeature pg = f.createPlot("pg_probe_fun");
      p("CREATEPLOT OK type=" + pg.getType());
      try { pg.run(); p("PLOT RUN OK"); } catch (Exception e) { p("PLOT RUN BAD " + e.getMessage()); }
    } catch (Exception e) {
      p("CREATEPLOT BAD " + e.getMessage());
    }

    try { m.save(OUTMPH); p("SAVE OK"); } catch (Exception e) { p("SAVE BAD " + e.getMessage()); }
  }
}
