import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeEvalTypes {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    String[] types = new String[]{"Average","AverageVolume","AvVolume","IntVolume","MaxVolume","IntSurface","MaxSurface","AvSurface"};
    for (String t: types) {
      try {
        try { m.result().numerical().remove("n1"); } catch (Exception e) {}
        m.result().numerical().create("n1", t);
        System.out.println("OKTYPE " + t);
      } catch (Exception e) {
        System.out.println("BADTYPE " + t + " :: " + e.getMessage());
      }
    }
  }
}
