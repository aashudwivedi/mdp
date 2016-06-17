package burlap.assignment4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MedianFinder {
	
	public static void main(String[] args) {
		
//		        int n=3;
//		        List<Integer> xPresent=Arrays.asList(-70, 110);
//		        List<Integer> yPresent=Arrays.asList(32, -240);
//		        int xMissing=1;
//		        int yMissing=1;
//		        int k=115;
		
//        int n=9;
//        List<Integer> xPresent=Arrays.asList(-167, -204, 195, 255, -206, -135, 165, 239);
//        List<Integer> yPresent=Arrays.asList(89, -141, 77, 133, -106, 85, -78, 91);
//        int xMissing=3;
//        int yMissing=5;
//        int k=44;
		
        int n=12;
        List<Integer> xPresent=Arrays.asList(-71, -159, -101, 171, -9, -215, 205, -161, -60, -117, 72);
        List<Integer> yPresent=Arrays.asList(-216, -129, -186, -27, -24, 48, 88, -85, -56, -128, -27);
        int xMissing=6;
        int yMissing=9;
        int k=23;
        
		        
		        
		        int limitMinX = -500;
		        int limitMaxX =  500;
		        
		        int limitMinY = -500;
		        int limitMaxY =  500;
		        
		        
		        List<Params> ans = new ArrayList<Params>();
		        
		        for(int inpX = limitMinX ; inpX <= limitMaxX ; inpX++)
		        {
		        	
		        	for(int inpY = limitMinY ; inpY <= limitMaxY ; inpY++)
		        	{
		        		List<Integer> xNew=new ArrayList<Integer>();
		        		List<Integer> yNew=new ArrayList<Integer>();
		           		List<Integer> xNew_unsorted=new ArrayList<Integer>();
		        		List<Integer> yNew_unsorted=new ArrayList<Integer>();
		        		
		        		xNew.addAll(xPresent);
		        		yNew.addAll(yPresent);
		        		xNew.add(xMissing, inpX);
		        		yNew.add(yMissing, inpY);
		        		xNew_unsorted.addAll(xNew);
		        		yNew_unsorted.addAll(yNew);
		        		Collections.sort(xNew);
		        		Collections.sort(yNew);
		        		
		        		//System.out.println("xNew>>" + xNew);
		        		//System.out.println("YNew>>" + yNew);
		        		
		   		        int xMedian = getMedian(xNew,n);
		   		        int yMedian = getMedian(yNew,n);
		   		        
		   		      //  System.out.println("xMedian" + xMedian);
		   		        //System.out.println("yMedian" + yMedian);
		        		
		   		        if(Math.abs(xMedian-yMedian) == k)
		   		        {
		   		        	
//		   		        	System.out.println("inpX>>" + inpX);
//		   		        	System.out.println("inpY>>" + inpY);
//		   		            System.out.println("xMedian" + xMedian);
//			   		        System.out.println("yMedian" + yMedian);
//			   		        System.out.println("Diff >>" + (xMedian-yMedian));
//			   		        System.out.println("Abs Diff >>" + Math.abs(xMedian-yMedian));
			        		
		   		        	
		   		        	
		   		        	int lInfinityDistance = getInfinityDistance(xNew_unsorted,yNew_unsorted);
		   		        	System.out.println("breaking @lInfinityDistance "+ lInfinityDistance + " bestX: " + inpX + " inpY: " + inpY);
//		   		        	System.out.println("lInfinityDistance>>" + lInfinityDistance); 
//		   		        	System.out.println("xNew_unsorted>>"+ xNew_unsorted);
//		   		        	System.out.println("yNew_unsorted>>"+ yNew_unsorted);
		   		        	if(ans.size() > 0 && ans.get(0) !=null && ans.get(0).getlDistance() > lInfinityDistance)
		   		        	{
		   		        		ans.remove(0);
		   		        	    ans.add(0, new Params(inpX, inpY, lInfinityDistance));
		   		        	}
		   		        	
		   		        	else if (ans.size() == 0)
		   		        	{
		   		        		
		   		        		ans.add(0, new Params(inpX, inpY, lInfinityDistance));
		   		        	}
		   		        	
		   		        	
		   		        	//System.exit(0);
		   		        	
		   		        }
		   		        
		   		        else {
		   		        	
		   		        	//System.out.println("trying>>" + Math.abs(xMedian-yMedian) + "  wheras k="+ k + "  inpX>>"+ inpX +" inpY>>"+inpY);
		   		        }
		        		
		        		
		        	}
		        	
		        	
		        	
		        }
		        
		       
		        System.out.println("Final ans >>" + ans.get(0));
		        System.out.println("bestX="+ans.get(0).getBestX());
		        System.out.println("bestY="+ans.get(0).getBestY());
		        System.out.println("LInfinityDistance="+ans.get(0).getlDistance());
		        

	}

	private static int getInfinityDistance(List<Integer> xNew, List<Integer> yNew) {
		
		List<Integer> diffList = new ArrayList<>();
		for (int i=0 ; i < xNew.size() ; i++)
		{
			diffList.add(Math.abs(xNew.get(i) - yNew.get(i)));
			
		}
		
	//	System.out.println("diffList>>" + diffList);
		Collections.sort(diffList);
		return diffList.get(diffList.size()-1);
		

	}

	private static int getMedian(List<Integer> xNew, int n) {
		
		int medianIndex = (n-1) / 2 ;
		
		return xNew.get(medianIndex);
		
		
	}

}


class Params
{
	public Params(int bestX, int bestY, int lDistance) {
		super();
		this.bestX = bestX;
		this.bestY = bestY;
		this.lDistance = lDistance;
	}
	public int getBestX() {
		return bestX;
	}
	public void setBestX(int bestX) {
		this.bestX = bestX;
	}
	public int getBestY() {
		return bestY;
	}
	public void setBestY(int bestY) {
		this.bestY = bestY;
	}
	public int getlDistance() {
		return lDistance;
	}
	public void setlDistance(int lDistance) {
		this.lDistance = lDistance;
	}
	int bestX ;
	int bestY ;
	int lDistance ;
	
	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return "bestX: " + bestX + " bestY: "+ bestY + " lDistance: "+lDistance;
	}
	
	
}


