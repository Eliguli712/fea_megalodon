import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AddStrict3BdfMeshViews {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) {
      if (needle.equals(t)) return true;
    }
    return false;
  }

  private static String safeMsg(Throwable t) {
    if (t == null) return "";
    String m = t.getMessage();
    if (m == null || m.isEmpty()) return t.getClass().getSimpleName();
    return m.replace('\n', ' ').replace('\r', ' ');
  }

  private static void safeSetCompView(Model model, String compTag, String viewTag, String key, String value) {
    try {
      model.component(compTag).view(viewTag).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static void ensureCompView(
      Model model, String compTag, String viewTag, String label, List<String> logs) {
    try {
      if (!hasTag(model.component(compTag).view().tags(), viewTag)) {
        model.component(compTag).view().create(viewTag, 3);
      }
      try {
        model.component(compTag).view(viewTag).label(label);
      } catch (Exception ignored) {
      }
      safeSetCompView(model, compTag, viewTag, "locked", "off");
      safeSetCompView(model, compTag, viewTag, "showgrid", "off");
      safeSetCompView(model, compTag, viewTag, "showaxis", "off");
      safeSetCompView(model, compTag, viewTag, "projection", "perspective");
      safeSetCompView(model, compTag, viewTag, "rendermesh", "on");
      safeSetCompView(model, compTag, viewTag, "transparency", "off");
      logs.add("VIEW_SETUP|" + compTag + "/" + viewTag + "|ok=true");
    } catch (Exception e) {
      logs.add("VIEW_SETUP|" + compTag + "/" + viewTag + "|ok=false|err=" + safeMsg(e));
    }
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
    logs.add("COMPONENTS|" + Arrays.toString(model.component().tags()));

    if (hasTag(model.component().tags(), "comp1")) {
      ensureCompView(model, "comp1", "view_mesh1_hr", "comp1 mesh1 high-res view", logs);
      ensureCompView(model, "comp1", "view_mesh2_hr", "comp1 mesh2 high-res view", logs);
      try {
        logs.add("COMP1_VIEWS|" + Arrays.toString(model.component("comp1").view().tags()));
      } catch (Exception e) {
        logs.add("COMP1_VIEWS|ERR|" + safeMsg(e));
      }
    } else {
      logs.add("VIEW_SETUP|comp1|ok=false|reason=missing_component");
    }

    if (hasTag(model.component().tags(), "comp2")) {
      ensureCompView(model, "comp2", "view_mesh3_hr", "comp2 mesh3 high-res view", logs);
      try {
        logs.add("COMP2_VIEWS|" + Arrays.toString(model.component("comp2").view().tags()));
      } catch (Exception e) {
        logs.add("COMP2_VIEWS|ERR|" + safeMsg(e));
      }
    } else {
      logs.add("VIEW_SETUP|comp2|ok=false|reason=missing_component");
    }

    model.save(MPH);
    logs.add("SAVED|" + MPH);

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
