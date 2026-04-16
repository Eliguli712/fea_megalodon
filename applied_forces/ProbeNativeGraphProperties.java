import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;

public class ProbeNativeGraphProperties {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String OUTMPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeNativeGraphProperties.out.mph";

  private static void p(String s) { System.out.println(s); }

  private static void trySet(PropFeature f, String key, Object value) {
    try {
      if (value instanceof String) f.set(key, (String) value);
      else if (value instanceof String[]) f.set(key, (String[]) value);
      else if (value instanceof Integer) f.set(key, ((Integer) value).intValue());
      else if (value instanceof int[]) f.set(key, (int[]) value);
      else if (value instanceof Boolean) f.set(key, ((Boolean) value).booleanValue());
      else if (value instanceof Double) f.set(key, ((Double) value).doubleValue());
      p("SET OK " + key + " = " + String.valueOf(value));
    } catch (Exception e) {
      p("SET BAD " + key + " = " + String.valueOf(value) + " :: " + e.getMessage());
    }
  }

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);

    try { m.result().remove("tbl_probe_native"); } catch (Exception ignored) {}
    try { m.result().remove("tg_probe_native"); } catch (Exception ignored) {}

    TableFeature tbl = m.result().table().create("tbl_probe_native", "Table");
    tbl.label("Probe Native Graph Table");
    tbl.setColumnHeaders(new String[]{"front_force_N", "trailing_force_N", "instant_impact_Wm2"});
    tbl.addRows(new double[][]{
      {500.0, 9366.115806784263, 107384.09760479114},
      {1000.0, 9440.063677720913, 110131.18297238447},
      {1500.0, 9514.15978598043, 112878.32001805623}
    });
    p("TABLE rows=" + tbl.getNRows());
    p("TABLE headers=" + Arrays.toString(tbl.getColumnHeaders()));
    p("TABLE real rows=" + Arrays.deepToString(tbl.getReal()));

    ResultFeature tg = m.result().create("tg_probe_native", "TableGraph");
    tg.label("Probe Root TableGraph");
    p("TYPE=" + tg.getType());

    String[] props = tg.properties();
    Arrays.sort(props);
    p("PROP COUNT=" + props.length);
    for (String key : props) {
      String vt = "";
      String[] allowed = null;
      try { vt = tg.getValueType(key); } catch (Exception e) { vt = "<err:" + e.getMessage() + ">"; }
      try { allowed = tg.getAllowedPropertyValues(key); } catch (Exception ignored) {}
      if (key.contains("table") || key.contains("col") || key.contains("data") || key.contains("legend") || key.contains("line") || key.contains("expr") || key.contains("unit") || key.contains("descr") || key.contains("x") || key.contains("y")) {
        p("PROP " + key + " type=" + vt + " allowed=" + (allowed == null ? "null" : Arrays.toString(allowed)));
      }
    }

    for (String key : new String[]{"table","xcol","ycol","xdata","legend","legendmethod","linecolor","linestyle","linewidth","expr","descr","unit"}) {
      try {
        p("KEY " + key + " type=" + tg.getValueType(key) + " allowed=" + Arrays.toString(tg.getAllowedPropertyValues(key)));
      } catch (Exception e) {
        p("KEY " + key + " <err> " + e.getMessage());
      }
    }

    trySet(tg, "table", "tbl_probe_native");
    trySet(tg, "xdata", "col");
    trySet(tg, "xcol", 1);
    trySet(tg, "ycol", new int[]{2});
    trySet(tg, "legend", true);
    trySet(tg, "linewidth", 3);
    trySet(tg, "linecolor", "red");
    trySet(tg, "descr", new String[]{"Trailing force"});
    trySet(tg, "unit", new String[]{"N"});

    try {
      tg.run();
      p("RUN OK");
    } catch (Exception e) {
      p("RUN BAD " + e.getMessage());
    }

    try {
      m.save(OUTMPH);
      p("SAVE OK " + OUTMPH);
    } catch (Exception e) {
      p("SAVE BAD " + e.getMessage());
    }
  }
}
