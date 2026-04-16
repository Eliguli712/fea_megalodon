import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

public class ProbeResultExportTypes {
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    System.out.println("EXPORT_TAGS=" + Arrays.toString(m.result().export().tags()));
    for (String t : m.result().export().tags()) {
      try {
        ExportFeature rf = m.result().export(t);
        System.out.println("EXPORT|" + t + "|type=" + rf.getType());
      } catch (Exception e) {
        System.out.println("EXPORT|" + t + "|ERR=" + e.getMessage());
      }
    }
  }
}
