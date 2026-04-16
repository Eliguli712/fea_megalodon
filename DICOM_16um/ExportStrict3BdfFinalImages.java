import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class ExportStrict3BdfFinalImages {
  private static final String MPH = "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static final String[] EXPORT_TAGS =
      new String[] {
        "img_std1",
        "img_std_nh",
        "img_std_og",
        "img_std_mr2",
        "img_std_mr5",
        "img_std_pr",
        "img_hr_pg_holo_std1",
        "img_hr_pg_holo_std_nh",
        "img_hr_pg_holo_std_og",
        "img_hr_pg_holo_std_mr2",
        "img_hr_pg_holo_std_mr5",
        "img_hr_pg_holo_std_pr",
        "img_hr_pg_vms_holocastic"
      };

  private static String safeMsg(Throwable t) {
    if (t == null) return "";
    String m = t.getMessage();
    if (m == null || m.isEmpty()) return t.getClass().getSimpleName();
    return m.replace('\n', ' ').replace('\r', ' ');
  }

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) {
      if (needle.equals(t)) return true;
    }
    return false;
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    int ok = 0;
    int fail = 0;

    for (String tag : EXPORT_TAGS) {
      if (!hasTag(model.result().export().tags(), tag)) {
        System.out.println("EXPORT_RUN|tag=" + tag + "|ok=false|err=missing_tag");
        fail++;
        continue;
      }

      try {
        model.result().export(tag).run();
        String fn = "";
        try {
          fn = model.result().export(tag).getString("pngfilename");
        } catch (Exception ignored) {
        }
        System.out.println("EXPORT_RUN|tag=" + tag + "|ok=true|file=" + fn);
        ok++;
      } catch (Exception e) {
        System.out.println("EXPORT_RUN|tag=" + tag + "|ok=false|err=" + safeMsg(e));
        fail++;
      }
    }

    System.out.println("EXPORT_SUMMARY|ok=" + ok + "|fail=" + fail + "|total=" + EXPORT_TAGS.length);
  }
}
