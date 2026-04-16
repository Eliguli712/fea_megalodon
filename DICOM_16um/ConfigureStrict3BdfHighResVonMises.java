import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ConfigureStrict3BdfHighResVonMises {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution.mph";
  private static final String OUT_DIR =
      "DICOM_16um/exports/highres_von_mises";

  private static final String BAD_EXPR = "comp1.solid.mises";
  private static final String SAFE_EXPR = "solid.mises";

  private static String ts() {
    return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
  }

  private static String safeType(ResultFeature rf) {
    try {
      return rf.getType();
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

  private static String sanitizeTag(String v) {
    return v.replaceAll("[^A-Za-z0-9_]", "_");
  }

  private static int normalizeExpr(PropFeature pf, String key, List<String> logs, String where) {
    int changed = 0;
    try {
      String[] arr = pf.getStringArray(key);
      if (arr != null && arr.length > 0) {
        boolean any = false;
        for (int i = 0; i < arr.length; i++) {
          String s = arr[i];
          if (s != null && s.contains(BAD_EXPR)) {
            String n = s.replace(BAD_EXPR, SAFE_EXPR);
            arr[i] = n;
            any = true;
            changed++;
            logs.add("PATCH|" + where + "|" + key + "[" + i + "]|" + s + " -> " + n);
          }
        }
        if (any) {
          pf.set(key, arr);
        }
      }
    } catch (Exception ignored) {
    }
    try {
      String s = pf.getString(key);
      if (s != null && s.contains(BAD_EXPR)) {
        String n = s.replace(BAD_EXPR, SAFE_EXPR);
        pf.set(key, n);
        changed++;
        logs.add("PATCH|" + where + "|" + key + "|" + s + " -> " + n);
      }
    } catch (Exception ignored) {
    }
    return changed;
  }

  private static int normalizeFeatureProps(PropFeature pf, String where, List<String> logs) {
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
    for (String p : props) {
      changed += normalizeExpr(pf, p, logs, where);
    }
    return changed;
  }

  private static boolean isVonMisesSurface(ResultFeature surface) {
    String expr = safeString(surface, "expr");
    if (expr.contains("mises")) {
      return true;
    }
    try {
      String[] arr = surface.getStringArray("expr");
      if (arr != null) {
        for (String s : arr) {
          if (s != null && s.contains("mises")) {
            return true;
          }
        }
      }
    } catch (Exception ignored) {
    }
    return false;
  }

  private static boolean hasVonMisesSurface(ResultFeature pg, List<String> logs) {
    boolean yes = false;
    try {
      String[] kids = pg.feature().tags();
      if (kids != null) {
        for (String k : kids) {
          ResultFeature child = pg.feature(k);
          if (!"Surface".equals(safeType(child))) {
            continue;
          }
          if (isVonMisesSurface(child)) {
            yes = true;
          }
        }
      }
    } catch (Exception ignored) {
    }
    return yes;
  }

  private static void configureHighResImageExport(Model m, String pgTag, List<String> logs) {
    String exTag = "img_hr_" + sanitizeTag(pgTag);
    if (exTag.length() > 55) {
      exTag = exTag.substring(0, 55);
    }

    String out = OUT_DIR + "/" + sanitizeTag(pgTag) + ".png";

    try {
      m.result().export().remove(exTag);
    } catch (Exception ignored) {
    }

    m.result().export().create(exTag, "Image3D");
    m.result().export(exTag).set("plotgroup", pgTag);

    try { m.result().export(exTag).set("imagetype", "png"); } catch (Exception ignored) {}
    try { m.result().export(exTag).set("qualitylevel", "95"); } catch (Exception ignored) {}
    try { m.result().export(exTag).set("unit", "px"); } catch (Exception ignored) {}
    try { m.result().export(exTag).set("width", 2400); } catch (Exception ignored) {}
    try { m.result().export(exTag).set("height", 1600); } catch (Exception ignored) {}
    try { m.result().export(exTag).set("lockratio", "off"); } catch (Exception ignored) {}
    try { m.result().export(exTag).set("zoomextents", "on"); } catch (Exception ignored) {}

    m.result().export(exTag).set("pngfilename", out);
    logs.add("EXPORT_CFG|plot=" + pgTag + "|export=" + exTag + "|file=" + out + "|size=2400x1600");
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    logs.add("MODEL|" + MPH);

    int patchedExpr = 0;
    int exportCfgCount = 0;

    for (String pgTag : model.result().tags()) {
      ResultFeature pg;
      try {
        pg = model.result(pgTag);
      } catch (Exception e) {
        continue;
      }
      String type = safeType(pg);
      if (!type.startsWith("PlotGroup")) {
        continue;
      }

      patchedExpr += normalizeFeatureProps(pg, "result/" + pgTag, logs);

      try {
        String[] kids = pg.feature().tags();
        if (kids != null) {
          for (String k : kids) {
            try {
              patchedExpr += normalizeFeatureProps(pg.feature(k), "result/" + pgTag + "/" + k, logs);
            } catch (Exception ignored) {
            }
          }
        }
      } catch (Exception ignored) {
      }

      if (hasVonMisesSurface(pg, logs)) {
        configureHighResImageExport(model, pgTag, logs);
        exportCfgCount++;
      }
    }

    String backup = MPH + ".bak-" + ts();
    model.save(backup);
    logs.add("BACKUP|" + backup);

    model.save(MPH);
    logs.add("SAVED|" + MPH);

    logs.add("PATCHED_EXPRESSION_COUNT|" + patchedExpr);
    logs.add("HIGHRES_EXPORT_CONFIGURED_COUNT|" + exportCfgCount);
    logs.add("OUT_DIR|" + OUT_DIR);

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
