import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class DiagnoseStrict3BdfSelections {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  // Failing postprocessing expression in static-force solid dynamics plots.
  private static final String BAD_EXPR = "comp1.solid.mises";
  private static final String SAFE_EXPR = "solid.mises";

  private static String nowTs() {
    return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
  }

  private static String safeMsg(Throwable t) {
    if (t == null) {
      return "";
    }
    String m = t.getMessage();
    if (m == null || m.isEmpty()) {
      return t.getClass().getSimpleName();
    }
    return m.replace('\n', ' ').replace('\r', ' ');
  }

  private static String safeType(ResultFeature rf) {
    try {
      return rf.getType();
    } catch (Exception ignored) {
      return "";
    }
  }

  private static String safeLabel(ResultFeature rf) {
    try {
      return rf.label();
    } catch (Exception ignored) {
      return "";
    }
  }

  private static String safeString(PropFeature pf, String key) {
    try {
      String v = pf.getString(key);
      return v == null ? "" : v;
    } catch (Exception ignored) {
      return "";
    }
  }

  private static int patchStringArrayProperty(PropFeature pf, String key, String where, List<String> logs) {
    int changed = 0;
    try {
      String[] arr = pf.getStringArray(key);
      if (arr == null || arr.length == 0) {
        return 0;
      }
      boolean any = false;
      for (int i = 0; i < arr.length; i++) {
        String v = arr[i];
        if (v != null && v.contains(BAD_EXPR)) {
          String nv = v.replace(BAD_EXPR, SAFE_EXPR);
          logs.add("PATCH|" + where + "|" + key + "[" + i + "]|" + v + " -> " + nv);
          arr[i] = nv;
          any = true;
          changed++;
        }
      }
      if (any) {
        pf.set(key, arr);
      }
    } catch (Exception ignored) {
    }
    return changed;
  }

  private static int patchStringProperty(PropFeature pf, String key, String where, List<String> logs) {
    int changed = 0;
    try {
      String v = pf.getString(key);
      if (v != null && v.contains(BAD_EXPR)) {
        String nv = v.replace(BAD_EXPR, SAFE_EXPR);
        pf.set(key, nv);
        logs.add("PATCH|" + where + "|" + key + "|" + v + " -> " + nv);
        changed++;
      }
    } catch (Exception ignored) {
    }
    return changed;
  }

  private static int patchFeatureProperties(PropFeature pf, String where, List<String> logs) {
    int changed = 0;
    String[] props;
    try {
      props = pf.properties();
    } catch (Exception ignored) {
      return 0;
    }
    if (props == null) {
      return 0;
    }

    for (String key : props) {
      changed += patchStringArrayProperty(pf, key, where, logs);
      changed += patchStringProperty(pf, key, where, logs);
    }
    return changed;
  }

  private static int walkResultFeature(
      ResultFeature rf,
      String path,
      String parentData,
      List<String> logs,
      Set<String> touchedPlotGroups,
      String rootPlotTag
  ) {
    int changed = 0;

    String type = safeType(rf);
    String label = safeLabel(rf);
    String selfData = safeString(rf, "data");
    String effData = (selfData == null || selfData.isEmpty() || "parent".equals(selfData)) ? parentData : selfData;

    if ("Surface".equals(type)) {
      String expr = safeString(rf, "expr");
      String[] exprArray = null;
      try {
        exprArray = rf.getStringArray("expr");
      } catch (Exception ignored) {
      }
      logs.add(
          "SURFACE|" + path
              + "|label=" + label
              + "|data=" + (effData == null ? "" : effData)
              + "|expr=" + expr
              + "|exprArray=" + (exprArray == null ? "null" : Arrays.toString(exprArray))
      );
    }

    int local = patchFeatureProperties(rf, path, logs);
    if (local > 0 && rootPlotTag != null && !rootPlotTag.isEmpty()) {
      touchedPlotGroups.add(rootPlotTag);
    }
    changed += local;

    try {
      String[] kids = rf.feature().tags();
      if (kids != null) {
        for (String k : kids) {
          ResultFeature child = rf.feature(k);
          changed += walkResultFeature(
              child,
              path + "/" + k,
              effData,
              logs,
              touchedPlotGroups,
              rootPlotTag
          );
        }
      }
    } catch (Exception ignored) {
    }

    return changed;
  }

  private static int patchNumericalAndExports(Model m, List<String> logs) {
    int changed = 0;

    try {
      String[] ntags = m.result().numerical().tags();
      if (ntags != null) {
        for (String tag : ntags) {
          try {
            changed += patchFeatureProperties(m.result().numerical(tag), "numerical/" + tag, logs);
          } catch (Exception ignored) {
          }
        }
      }
    } catch (Exception ignored) {
    }

    try {
      String[] etags = m.result().export().tags();
      if (etags != null) {
        for (String tag : etags) {
          try {
            changed += patchFeatureProperties(m.result().export(tag), "export/" + tag, logs);
          } catch (Exception ignored) {
          }
        }
      }
    } catch (Exception ignored) {
    }

    return changed;
  }

  private static int countResultExprHits(ResultFeature rf) {
    int count = 0;

    try {
      String[] props = rf.properties();
      if (props != null) {
        for (String key : props) {
          try {
            String[] arr = rf.getStringArray(key);
            if (arr != null) {
              for (String v : arr) {
                if (v != null && v.contains(BAD_EXPR)) {
                  count++;
                }
              }
            }
          } catch (Exception ignored) {
          }
          try {
            String v = rf.getString(key);
            if (v != null && v.contains(BAD_EXPR)) {
              count++;
            }
          } catch (Exception ignored) {
          }
        }
      }
    } catch (Exception ignored) {
    }

    try {
      String[] kids = rf.feature().tags();
      if (kids != null) {
        for (String k : kids) {
          count += countResultExprHits(rf.feature(k));
        }
      }
    } catch (Exception ignored) {
    }

    return count;
  }

  private static int countRemainingHits(Model m) {
    int count = 0;

    try {
      for (String r : m.result().tags()) {
        try {
          count += countResultExprHits(m.result(r));
        } catch (Exception ignored) {
        }
      }
    } catch (Exception ignored) {
    }

    try {
      String[] ntags = m.result().numerical().tags();
      if (ntags != null) {
        for (String tag : ntags) {
          try {
            String[] props = m.result().numerical(tag).properties();
            if (props == null) {
              continue;
            }
            for (String key : props) {
              try {
                String[] arr = m.result().numerical(tag).getStringArray(key);
                if (arr != null) {
                  for (String v : arr) {
                    if (v != null && v.contains(BAD_EXPR)) {
                      count++;
                    }
                  }
                }
              } catch (Exception ignored2) {
              }
              try {
                String v = m.result().numerical(tag).getString(key);
                if (v != null && v.contains(BAD_EXPR)) {
                  count++;
                }
              } catch (Exception ignored2) {
              }
            }
          } catch (Exception ignored) {
          }
        }
      }
    } catch (Exception ignored) {
    }

    return count;
  }

  private static double evalBoundary1Mises(Model m, String datasetTag, String expr) {
    final String numTag = "diag_mises_b1";
    try {
      try {
        m.result().numerical().remove(numTag);
      } catch (Exception ignored) {
      }
      m.result().numerical().create(numTag, "MaxSurface");
      m.result().numerical(numTag).set("data", datasetTag);
      m.result().numerical(numTag).set("expr", new String[]{expr});
      m.result().numerical(numTag).selection().geom("geom1", 2);
      m.result().numerical(numTag).selection().set(new int[]{1});
      m.result().numerical(numTag).setResult();
      double[][] r = m.result().numerical(numTag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        return r[0][0];
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static void validateTouchedPlots(Model m, Set<String> touchedPlotGroups, List<String> logs) {
    for (String pg : touchedPlotGroups) {
      try {
        m.result(pg).run();
        logs.add("PLOT_VALIDATE|" + pg + "|ok=true");
      } catch (Exception e) {
        logs.add("PLOT_VALIDATE|" + pg + "|ok=false|error=" + safeMsg(e));
      }
    }
  }

  private static boolean hasSurfaceChild(ResultFeature rf) {
    try {
      String[] kids = rf.feature().tags();
      if (kids == null) {
        return false;
      }
      for (String k : kids) {
        try {
          if ("Surface".equals(rf.feature(k).getType())) {
            return true;
          }
        } catch (Exception ignored) {
        }
      }
    } catch (Exception ignored) {
    }
    return false;
  }

  private static int forceSurfaceExprFix(ResultFeature rf, String path, List<String> logs) {
    int n = 0;
    try {
      String[] kids = rf.feature().tags();
      if (kids == null) {
        return 0;
      }
      for (String k : kids) {
        ResultFeature child = rf.feature(k);
        String type = "";
        try { type = child.getType(); } catch (Exception ignored) {}
        if (!"Surface".equals(type)) {
          continue;
        }
        String expr = safeString(child, "expr");
        if (expr.contains(BAD_EXPR)) {
          String nv = expr.replace(BAD_EXPR, SAFE_EXPR);
          try {
            child.set("expr", nv);
            n++;
            logs.add("FORCE_PATCH|" + path + "/" + k + "|expr|" + expr + " -> " + nv);
          } catch (Exception ignored) {
          }
        }
      }
    } catch (Exception ignored) {
    }
    return n;
  }

  private static int validateAllSurfacePlots(Model m, List<String> logs) {
    int fixed = 0;
    for (String pg : m.result().tags()) {
      ResultFeature rf;
      try {
        rf = m.result(pg);
      } catch (Exception e) {
        continue;
      }
      String rootType = "";
      try { rootType = rf.getType(); } catch (Exception ignored) {}
      if (!rootType.startsWith("PlotGroup")) {
        continue;
      }
      if (!hasSurfaceChild(rf)) {
        continue;
      }

      try {
        rf.run();
        logs.add("PLOT_RUN|" + pg + "|ok=true");
      } catch (Exception e) {
        String msg = safeMsg(e);
        logs.add("PLOT_RUN|" + pg + "|ok=false|error=" + msg);
        if (msg.contains(BAD_EXPR)) {
          int n = forceSurfaceExprFix(rf, "result/" + pg, logs);
          fixed += n;
          if (n > 0) {
            try {
              rf.run();
              logs.add("PLOT_RERUN|" + pg + "|ok=true");
            } catch (Exception e2) {
              logs.add("PLOT_RERUN|" + pg + "|ok=false|error=" + safeMsg(e2));
            }
          }
        }
      }
    }
    return fixed;
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    Set<String> touchedPlots = new LinkedHashSet<String>();

    logs.add("MODEL|" + MPH);
    logs.add("RESULT_TAGS|" + Arrays.toString(model.result().tags()));

    int patched = 0;
    for (String rootTag : model.result().tags()) {
      ResultFeature root;
      try {
        root = model.result(rootTag);
      } catch (Exception e) {
        logs.add("SKIP|result/" + rootTag + "|error=" + safeMsg(e));
        continue;
      }
      String rootType = safeType(root);
      String rootData = safeString(root, "data");

      if (rootType.startsWith("PlotGroup")) {
        patched += walkResultFeature(
            root,
            "result/" + rootTag,
            rootData,
            logs,
            touchedPlots,
            rootTag
        );
      }
    }

    patched += patchNumericalAndExports(model, logs);

    int remainingBeforeSave = countRemainingHits(model);
    logs.add("REMAINING_HITS_BEFORE_SAVE|" + remainingBeforeSave);

    // Explicit checks on static-force solid-dynamics dataset.
    double badEval = evalBoundary1Mises(model, "dset6", BAD_EXPR);
    double safeEval = evalBoundary1Mises(model, "dset6", SAFE_EXPR);
    logs.add("EVAL|dataset=dset6|expr=" + BAD_EXPR + "|value=" + badEval + "|finite=" + Double.isFinite(badEval));
    logs.add("EVAL|dataset=dset6|expr=" + SAFE_EXPR + "|value=" + safeEval + "|finite=" + Double.isFinite(safeEval));

    validateTouchedPlots(model, touchedPlots, logs);
    patched += validateAllSurfacePlots(model, logs);

    if (patched > 0) {
      String backup = MPH + ".bak-" + nowTs();
      model.save(backup);
      logs.add("BACKUP|" + backup);
      model.save(MPH);
      logs.add("SAVED|" + MPH);
    }

    logs.add("PATCHED_COUNT|" + patched);
    logs.add("TOUCHED_PLOTS|" + touchedPlots.toString());
    logs.add("REMAINING_HITS_FINAL|" + countRemainingHits(model));

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
