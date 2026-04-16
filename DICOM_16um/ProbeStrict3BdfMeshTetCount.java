import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeStrict3BdfMeshTetCount {
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    try {
      int t = m.component("comp1").mesh("mesh2").getNumElem("tet");
      System.out.println("COMP1_MESH2_TET|" + t);
    } catch (Exception e) {
      System.out.println("COMP1_MESH2_TET|ERR|" + e.getMessage());
    }
    try {
      int t = m.component("comp1").mesh("mesh1").getNumElem("tet");
      System.out.println("COMP1_MESH1_TET|" + t);
    } catch (Exception e) {
      System.out.println("COMP1_MESH1_TET|ERR|" + e.getMessage());
    }
    try {
      int t = m.mesh("mesh2").getNumElem("tet");
      System.out.println("GLOBAL_MESH2_TET|" + t);
    } catch (Exception e) {
      System.out.println("GLOBAL_MESH2_TET|ERR|" + e.getMessage());
    }
    try {
      int t = m.mesh("mpart2").getNumElem("tet");
      System.out.println("GLOBAL_MPART2_TET|" + t);
    } catch (Exception e) {
      System.out.println("GLOBAL_MPART2_TET|ERR|" + e.getMessage());
    }
  }
}
