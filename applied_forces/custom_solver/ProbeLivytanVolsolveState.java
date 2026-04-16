import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeLivytanVolsolveState {
  private static final String FILE1 =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";
  private static final String FILE2 =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_volsolve_out_Model.mph";

  private static void probe(String file) throws IOException {
    Model m = ModelUtil.load("Model", file);
    System.out.println("FILE|" + file);
    try {
      String[] comps = m.component().tags();
      System.out.println("COMPONENTS|" + String.join(",", comps));
    } catch (Exception e) {
      System.out.println("COMPONENTS|<none>");
    }
    try {
      String[] meshes = m.component("comp1").mesh().tags();
      System.out.println("MESHES|" + String.join(",", meshes));
    } catch (Exception e) {
      System.out.println("MESHES|<none>");
    }
    try {
      int ntet = m.component("comp1").mesh("mesh1").getNumElem("tet");
      int ntri = m.component("comp1").mesh("mesh1").getNumElem("tri");
      int npts = m.component("comp1").mesh("mesh1").getNumVertex();
      System.out.println("MESH_COUNTS|vertices=" + npts + "|tri=" + ntri + "|tet=" + ntet);
    } catch (Exception e) {
      System.out.println("MESH_COUNTS|<unavailable>|" + e.getMessage());
    }
    try {
      m.component("comp1").physics("solid").selection().all();
      int nd = m.component("comp1").physics("solid").selection().entities(3).length;
      int nb = m.component("comp1").physics("solid").selection().entities(2).length;
      System.out.println("SOLID_SELECTION|domains=" + nd + "|boundaries=" + nb);
    } catch (Exception e) {
      System.out.println("SOLID_SELECTION|<none>|" + e.getMessage());
    }
    try {
      int nfix = m.component("comp1").physics("solid").feature("fix1").selection().entities(2).length;
      System.out.println("FIX1_BOUNDARIES|" + nfix);
    } catch (Exception e) {
      System.out.println("FIX1_BOUNDARIES|<none>|" + e.getMessage());
    }
    try {
      String[] studies = m.study().tags();
      System.out.println("STUDIES|" + String.join(",", studies));
    } catch (Exception e) {
      System.out.println("STUDIES|<none>");
    }
    try {
      String[] dsets = m.result().dataset().tags();
      System.out.println("DATASETS|" + String.join(",", dsets));
    } catch (Exception e) {
      System.out.println("DATASETS|<none>");
    }
    try {
      String[] pgs = m.result().tags();
      System.out.println("PLOT_GROUPS|" + String.join(",", pgs));
    } catch (Exception e) {
      System.out.println("PLOT_GROUPS|<none>");
    }
  }

  public static void main(String[] args) throws Exception {
    probe(FILE1);
    probe(FILE2);
  }
}
