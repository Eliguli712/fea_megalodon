import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeHyperSelectedMises {
  private static int[] keptDomains() {
    int[] rem = new int[]{2,5,6,25,28,46,48,51,62,84,91,98,100,106,110,116,121,127,131,135,152,165,182};
    boolean[] keep = new boolean[184];
    for (int i=1;i<=183;i++) keep[i] = true;
    for (int r: rem) keep[r] = false;
    int n=0; for (int i=1;i<=183;i++) if (keep[i]) n++;
    int[] out = new int[n]; int k=0;
    for (int i=1;i<=183;i++) if (keep[i]) out[k++]=i;
    return out;
  }
  private static void eval(Model m, String tag, String dset) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).selection().geom("geom1",3);
      m.result().numerical(tag).selection().set(keptDomains());
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      System.out.println(tag + "=" + (r!=null&&r.length>0&&r[0].length>0 ? r[0][0] : Double.NaN));
    } catch (Exception e) {
      System.out.println(tag + " failed: " + e.getMessage());
    }
  }
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeBodyLoadHyper_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
    eval(m, "mx_nh_sel", "dset1");
    eval(m, "mx_og_sel", "dset2");
  }
}
