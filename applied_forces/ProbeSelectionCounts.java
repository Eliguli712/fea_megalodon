import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.Arrays;

public class ProbeSelectionCounts {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    String[] tags = m.component("comp1").selection().tags();
    Arrays.sort(tags);
    for (String t : tags) {
      String op = "";
      try { op = m.component("comp1").selection(t).getType(); } catch (Exception e) {}
      int n2 = -1, n3 = -1;
      try {
        int[] e2 = m.component("comp1").selection(t).entities(2);
        n2 = (e2 == null) ? -1 : e2.length;
      } catch (Exception e) {}
      try {
        int[] e3 = m.component("comp1").selection(t).entities(3);
        n3 = (e3 == null) ? -1 : e3.length;
      } catch (Exception e) {}
      if (n2 > 0 || n3 > 0) {
        System.out.println(t + " type=" + op + " dim2=" + n2 + " dim3=" + n3);
      }
    }
  }
}
