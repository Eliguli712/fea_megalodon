import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeLivytanMeshSource {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model", e);
    }

    try {
      String[] tags = m.component("comp1").mesh("mesh1").feature().tags();
      System.out.println("MESH_FEATURES|" + String.join(",", tags));
    } catch (Exception e) {
      System.out.println("MESH_FEATURES|<none>|" + e.getMessage());
    }

    try {
      String src = m.component("comp1").mesh("mesh1").feature("imp1").getString("source");
      String fn = m.component("comp1").mesh("mesh1").feature("imp1").getString("filename");
      System.out.println("IMP1_SOURCE|" + src);
      System.out.println("IMP1_FILENAME|" + fn);
    } catch (Exception e) {
      System.out.println("IMP1|<missing>|" + e.getMessage());
    }

    try {
      int nv = m.component("comp1").mesh("mesh1").getNumVertex();
      int ntri = m.component("comp1").mesh("mesh1").getNumElem("tri");
      int ntet = m.component("comp1").mesh("mesh1").getNumElem("tet");
      System.out.println("MESH_COUNTS|vertices=" + nv + "|tri=" + ntri + "|tet=" + ntet);
    } catch (Exception e) {
      System.out.println("MESH_COUNTS|<unavailable>|" + e.getMessage());
    }
  }
}
