import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeExportFiles {
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    for (String t : m.result().export().tags()) {
      if (!t.startsWith("img_")) continue;
      try {
        String fn = "";
        try { fn = m.result().export(t).getString("pngfilename"); } catch (Exception ignored) {}
        String data = "";
        try { data = m.result().export(t).getString("plotgroup"); } catch (Exception ignored) {}
        System.out.println("EXP|" + t + "|plot=" + data + "|file=" + fn);
      } catch (Exception e) {
        System.out.println("EXP|" + t + "|ERR=" + e.getMessage());
      }
    }
  }
}
