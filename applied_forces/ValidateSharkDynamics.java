import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ValidateSharkDynamics {
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

  private static double evalMaxSelected(Model m, String tag, String dset, int[] dom) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).selection().geom("geom1",3);
      m.result().numerical(tag).selection().set(dom);
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      return (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
    } catch (Exception e) {
      System.out.println("eval " + tag + " failed: " + e.getMessage());
      return Double.NaN;
    }
  }

  private static double evalMaxSurface(Model m, String tag, String dset) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, "MaxSurface");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      return (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
    } catch (Exception e) {
      System.out.println("eval " + tag + " failed: " + e.getMessage());
      return Double.NaN;
    }
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    boolean tor1Exists;
    try { m.geom("part1").feature("tor1"); tor1Exists = true; }
    catch (Exception e) { tor1Exists = false; }
    System.out.println("tor1_exists=" + tor1Exists);

    try {
      m.study("std1").run();
      System.out.println("std1_re_run=ok");
    } catch (Exception e) {
      System.out.println("std1_re_run=fail: " + e.getMessage());
    }

    int[] dom = keptDomains();
    double mxv = evalMaxSelected(m, "mx_std1_sel_chk", "dset6", dom);
    double mxs = evalMaxSurface(m, "mx_std1_surf_chk", "dset6");
    System.out.println("std1_max_mises_selected=" + mxv);
    System.out.println("std1_max_mises_surface=" + mxs);

    try {
      m.result("pg_vms_std1").run();
      System.out.println("pg_vms_std1=ok");
    } catch (Exception e) {
      System.out.println("pg_vms_std1=fail: " + e.getMessage());
    }
  }
}
