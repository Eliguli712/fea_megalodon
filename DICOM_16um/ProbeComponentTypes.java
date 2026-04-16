import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

public class ProbeComponentTypes {
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    System.out.println("COMPONENTS=" + Arrays.toString(m.component().tags()));
    for (String c : m.component().tags()) {
      try {
        System.out.println("COMP|" + c + "|type=" + m.component(c).getType());
      } catch (Exception e) {
        System.out.println("COMP|" + c + "|ERR=" + e.getMessage());
      }
      try {
        System.out.println("  GEOMS=" + Arrays.toString(m.component(c).geom().tags()));
      } catch (Exception e) {
        System.out.println("  GEOMS_ERR=" + e.getMessage());
      }
      try {
        System.out.println("  MESHES=" + Arrays.toString(m.component(c).mesh().tags()));
      } catch (Exception e) {
        System.out.println("  MESHES_ERR=" + e.getMessage());
      }
      try {
        System.out.println("  PHYS=" + Arrays.toString(m.component(c).physics().tags()));
      } catch (Exception e) {
        System.out.println("  PHYS_ERR=" + e.getMessage());
      }
    }
  }
}
