import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class RunLivytanStd12AndCollect {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";
  private static final String OUT_JSON =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_std12_results.json";

  private static void p(String s) { System.out.println(s); }

  private static boolean hasStudy(Model m, String tag) {
    try { m.study(tag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasResult(Model m, String tag) {
    try { m.result(tag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasResultFeature(Model m, String pg, String ft) {
    try { m.result(pg).feature(ft); return true; } catch (Exception e) { return false; }
  }

  private static double evalVolume(Model m, String tag, String type, String expr, String dataset) {
    try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
    try {
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).set("data", dataset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception e) {
      p("EVAL_WARN|" + tag + "|" + e.getMessage());
    }
    return Double.NaN;
  }

  private static void ensureVmsPlot(Model m, String pg, String label, String dset) {
    if (!hasResult(m, pg)) m.result().create(pg, "PlotGroup3D");
    try { m.result(pg).label(label); } catch (Exception ignored) {}
    try { m.result(pg).set("data", dset); } catch (Exception ignored) {}
    try { m.result(pg).set("view", "view1"); } catch (Exception ignored) {}

    if (!hasResultFeature(m, pg, "surf1")) m.result(pg).create("surf1", "Surface");
    try { m.result(pg).feature("surf1").set("expr", "solid.mises"); } catch (Exception ignored) {}
    try { m.result(pg).feature("surf1").set("unit", "Pa"); } catch (Exception ignored) {}
    try { m.result(pg).feature("surf1").selection().all(); } catch (Exception ignored) {}
    try { m.result(pg).run(); } catch (Exception e) { p("PLOT_WARN|" + pg + "|" + e.getMessage()); }
  }

  private static String jsonNum(double v) {
    return Double.isFinite(v) ? Double.toString(v) : "null";
  }

  public static void main(String[] args) {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model", e);
    }

    String backup = MPH.replace(
        ".mph",
        ".pre_std12run_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".mph"
    );
    try {
      m.save(backup);
      p("BACKUP|" + backup);
    } catch (IOException e) {
      p("BACKUP_WARN|" + e.getMessage());
    }

    if (!hasStudy(m, "std1") || !hasStudy(m, "std2")) {
      throw new RuntimeException("Missing required studies std1/std2");
    }

    p("RUN|std1");
    m.study("std1").run();
    p("DONE|std1");

    p("RUN|std2");
    m.study("std2").run();
    p("DONE|std2");

    String ds1 = "dset1";
    String ds2 = "dset2";

    double std1MaxVms = evalVolume(m, "mx_std1_vms", "MaxVolume", "solid.mises", ds1);
    double std1AvgVms = evalVolume(m, "av_std1_vms", "AvVolume", "solid.mises", ds1);
    double std1MaxDisp = evalVolume(m, "mx_std1_u", "MaxVolume", "sqrt(u^2+v^2+w^2)", ds1);
    double std1AvgDisp = evalVolume(m, "av_std1_u", "AvVolume", "sqrt(u^2+v^2+w^2)", ds1);

    double std2MaxVms = evalVolume(m, "mx_std2_vms", "MaxVolume", "solid.mises", ds2);
    double std2AvgVms = evalVolume(m, "av_std2_vms", "AvVolume", "solid.mises", ds2);
    double std2MaxDisp = evalVolume(m, "mx_std2_u", "MaxVolume", "sqrt(u^2+v^2+w^2)", ds2);
    double std2AvgDisp = evalVolume(m, "av_std2_u", "AvVolume", "sqrt(u^2+v^2+w^2)", ds2);

    ensureVmsPlot(m, "pg_std1_vms", "std1 Von Mises", ds1);
    ensureVmsPlot(m, "pg_std2_vms", "std2 Von Mises", ds2);

    try {
      m.save(MPH);
      p("SAVED|" + MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model", e);
    }

    String ts = LocalDateTime.now().toString();
    String json = "{\n"
        + "  \"timestamp\": \"" + ts + "\",\n"
        + "  \"studies\": [\"std1\", \"std2\"],\n"
        + "  \"datasets\": {\"std1\": \"" + ds1 + "\", \"std2\": \"" + ds2 + "\"},\n"
        + "  \"std1\": {\n"
        + "    \"max_mises_pa\": " + jsonNum(std1MaxVms) + ",\n"
        + "    \"avg_mises_pa\": " + jsonNum(std1AvgVms) + ",\n"
        + "    \"max_disp_m\": " + jsonNum(std1MaxDisp) + ",\n"
        + "    \"avg_disp_m\": " + jsonNum(std1AvgDisp) + "\n"
        + "  },\n"
        + "  \"std2\": {\n"
        + "    \"max_mises_pa\": " + jsonNum(std2MaxVms) + ",\n"
        + "    \"avg_mises_pa\": " + jsonNum(std2AvgVms) + ",\n"
        + "    \"max_disp_m\": " + jsonNum(std2MaxDisp) + ",\n"
        + "    \"avg_disp_m\": " + jsonNum(std2AvgDisp) + "\n"
        + "  }\n"
        + "}\n";

    try (FileWriter fw = new FileWriter(OUT_JSON)) {
      fw.write(json);
    } catch (IOException e) {
      throw new RuntimeException("Failed to write JSON", e);
    }

    p("RESULTS_JSON|" + OUT_JSON);
    p("METRIC|std1|max_mises=" + std1MaxVms + "|avg_mises=" + std1AvgVms + "|max_disp=" + std1MaxDisp + "|avg_disp=" + std1AvgDisp);
    p("METRIC|std2|max_mises=" + std2MaxVms + "|avg_mises=" + std2AvgVms + "|max_disp=" + std2MaxDisp + "|avg_disp=" + std2AvgDisp);
  }
}
