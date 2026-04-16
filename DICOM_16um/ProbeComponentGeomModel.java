import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeComponentGeomModel {
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    for (String c : m.component().tags()) {
      try {
        System.out.println("COMP|" + c + "|type=" + m.component(c).getType() + "|geometricModel=" + m.component(c).geometricModel());
      } catch (Exception e) {
        System.out.println("COMP|" + c + "|ERR=" + e.getMessage());
      }
    }
  }
}
