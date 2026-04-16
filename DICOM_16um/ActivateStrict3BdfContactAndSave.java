import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;

public class ActivateStrict3BdfContactAndSave {
  private static final String MPH = "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";
  private static final String[] CONTACT_FEATURES = new String[] {"dcnt1", "dgcnt1"};

  private static String ts() {
    return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
  }

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

  private static String activeState(Object feature) {
    try {
      Method m = feature.getClass().getMethod("isActive");
      return String.valueOf(m.invoke(feature));
    } catch (Exception ignored) {
      return "unknown";
    }
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    String[] solidTags = model.component("comp1").physics("solid").feature().tags();
    System.out.println("SOLID_FEATURES|" + Arrays.toString(solidTags));

    for (String ctag : CONTACT_FEATURES) {
      if (!hasTag(solidTags, ctag)) {
        System.out.println("CONTACT_SET|feature=" + ctag + "|ok=false|reason=missing");
        continue;
      }

      try {
        model.component("comp1").physics("solid").feature(ctag).active(true);
      } catch (Exception e) {
        System.out.println("CONTACT_SET|feature=" + ctag + "|ok=false|stage=active|err=" + safeMsg(e));
        continue;
      }

      try {
        model.component("comp1").physics("solid").feature(ctag).set("pairSelection", "all");
      } catch (Exception ignored) {
      }
      try {
        model.component("comp1").physics("solid").feature(ctag).set("StudyStep", "std1/stat");
      } catch (Exception ignored) {
      }

      String state;
      try {
        state = activeState(model.component("comp1").physics("solid").feature(ctag));
      } catch (Exception e) {
        state = "unknown";
      }
      System.out.println("CONTACT_SET|feature=" + ctag + "|ok=true|active=" + state + "|step=std1/stat");
    }

    String backup = MPH + ".bak-" + ts();
    model.save(backup);
    model.save(MPH);
    System.out.println("SAVED|" + MPH);
    System.out.println("BACKUP|" + backup);
  }
}
