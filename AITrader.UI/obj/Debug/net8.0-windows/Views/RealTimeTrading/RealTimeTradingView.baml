   M S B A M L   `   `   `  ����F  BAITrader.UI, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null_4clr-namespace:AITrader.UI.ViewModels.RealTimeTrading&AITrader.UI.ViewModels.RealTimeTrading  5      ?$clr-namespace:AITrader.UI.ConvertersAITrader.UI.Converters  5      ;    5AITrader.UI.Views.RealTimeTrading.RealTimeTradingView   5      W SPresentationCore, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35F 9http://schemas.microsoft.com/winfx/2006/xaml/presentation     5      R NSystem.Xaml, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5      Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5      @d2http://schemas.microsoft.com/expression/blend/2008     5      _ [MaterialDesignThemes.Wpf, Version=4.9.0.0, Culture=neutral, PublicKeyToken=df2a72020bd7962aFmaterialDesign1http://materialdesigninxaml.net/winfx/xaml/themes  5      E
viewModels4clr-namespace:AITrader.UI.ViewModels.RealTimeTrading  5      5
converters$clr-namespace:AITrader.UI.Converters  5        � 	Resources  5         	BoolToVis4   .AITrader.UI.Converters.BooleanToColorConverter  BooleanToColorConverter  ConnectionIndicator �� 
TargetType �� Property �� Value �� Triggers �� Binding �� Value  PositionTextStyle��%G  &	        &	      &	 &     &	 �    �� 5   
   6'     5   
   6-   �� 5   
   " A�6,   .�5      ��  � ��$ 12��6&   �� 5       � ��$ 12��6'   �� 5       � ��$ 5,0,5,0q�6'    5      �� 5       6   �� *IsDataConnected+ True6B   .a�5      ��  � ��
 � � �6-   5      �� 5       6   �� *IsDataConnected+	 False6B   .a�5      ��  � ��
 �  ��6-   5      5      �� 5   
   " ��6*    5       �� 5!       6   �� *CurrentPosition+ 16B   .a�5"      ��  � ��
 � � �63   5#      �� 5$       6   �� *CurrentPosition+ -16B   .a�5%      ��  � ��
 �  ��63   5&      �� 5'       6   �� *CurrentPosition+ 06B   .a�5(      ��  � ��
 �����63   5)      5+      5,      .��5.      �  � RowDefinitions 5/   
   �� 50      $
��Auto �6   �� 51      $��* �6   .+�55   
   #  MaterialDesignThemes.Wpf.Card  $��0��6   $��10q�6+   $��10q�67   .��56      �  � ColumnDefinitions 57      �� 58      $��* �6'   �� 59      $
��Auto �6'   .+�5<      �� $��0��6   	 �� Orientation$	 
Horizontal=�6-   $��Center=�6F   .+�5=      �� $��Real-Time Trading��6     $  MaterialDesignHeadline5TextBlock#��[ 69     5>      $��15,0,0,0q�6*   $��	10,5,10,5q�6<   ���#f5f5f5.��5?      �� $	 
Horizontal=�6%   .+�5@      �� $��Data: ��6(   $��Center=�66   A� 5A      #��[ 6&   �� 5B      $��Order: ��6(   $��Center=�67   $��10,0,0,0q�6R   A� 5C      #��[ 6&   ��5D   &   �� *IsOrderConnected+
 �� 	Converter#
 [  �� ConverterParameter$ 	Green:Red���� 5E      $��
Position: ��6(   $��Center=�6:   $��10,0,0,0q�6U   �� 5F      $��Center=�6M   #��[ 5G   (   $
��Bold"�6S   ��5F   (   �� *CurrentPositionText+5G   S   5H      5I      5J      �� 5L      $��1��6   $	 
Horizontal=�6-   .+�5M      ��   MaterialDesignRaisedButton#��[ 5N      $��5q�5O       � ToolTip$ Start real-time trading��5P      ��5M      �� *StartTradingCommand+.��5Q      �� $	 
Horizontal=�6%   .+�5R      '  !MaterialDesignThemes.Wpf.PackIcon     Kind$
 Play=�66   $��0,0,5,0q�6B   �� 5S      $��Start��6(   5T      5U      �� 5W         MaterialDesignOutlinedButton#��[ 5X      $��5q�5Y      $ Stop real-time trading��5Z      ��5W      �� *StopTradingCommand+.��5[      �� $	 
Horizontal=�6%   .+�5\        $
 Stop=�66   $��0,0,5,0q�6B   �� 5]      $
��Stop��6(   5^      5_      5`      5a      5b      � 5e   
   $��1��6   $��10q�6    5f      �� 5g      $��* �6#   .+�5k        $��10q�6"   .��5l      �  5m      �� 5n      $
��Auto �6(   �� 5o      $��* �6(   .+�5s      �� $��0��6!   .+�5t      �� $��Trading Parameters��6$    $  MaterialDesignHeadline6TextBlock#��[ 5u   $   $��0,0,0,10q�5v   $   �� 5x      $��Enable Trading��6#   $��0,5,0,5q�5z   #    d� 	IsChecked 5y   #   �� *TradingEnabled+5z   #   �� 5|      $��Position Sizing��6$   $��0,10,0,5q�6;   �� 5}       � Minimum$	 0.1Q�5~   !    � Maximum$	 5.0Q�6/    �� TickFrequency$	 0.1Q�5   !    �� IsSnapToTickEnabled$
 True��5�   !    � Value 5}   !   �� *PositionSizing+5�   !   �� 5�      $��Right=�5�   $   ��5�   $   �� *PositionSizing+ �� StringFormat$ N1��5�   $   �� 5�      $��Stop Loss (Ticks)��6$   $��0,10,0,5q�6=   �� 5�      $ 1Q�5�   !   $ 50Q�6-   $ 1Q�5�   !   $
 True��5�   !    5�   !   �� *StopLossTicks+5�   !   �� 5�      $��Right=�5�   $   ��5�   $   �� *StopLossTicks+5�   $   �� 5�      $��Take Profit (Ticks)��6$   $��0,10,0,5q�6?   �� 5�      $ 1Q�5�   !   $	 100Q�6-   $ 1Q�5�   !   $
 True��5�   !    5�   !   �� *TakeProfitTicks+5�   !   �� 5�      $��Right=�5�   $   ��5�   $   �� *TakeProfitTicks+5�   $   �� 5�      $��Update Parameters��5�   !   #��[ 5�   !   $��0,15,0,5q�5�   !   ��5�   !   �� *UpdateParametersCommand+5�   !   5�        5�      $��1��6*   $��10q�67   $��0,10,0,0q�6D   $��Stretch=�5�   *   .��5�      �� $
��Auto=�6'   .��5�      �� ��6+   �� *StatusMessages+��5�   "   �� 5�   &   .R�5�   *   �� $
��Wrap=�5�   4   $��0,2,0,2q�5�   4   ��5�   4   �� 5�   4   5�   '   5�      5�      5�      5�      5�      5�      � 5�   
   $��0��6   $��2��6   ���	#80000000y�5�      �� *	IsLoading+#
 [  .+�5�      �� $��Center=�6   $��Center=�64   .+�5�      �  % !MaterialDesignCircularProgressBar#��[ 6   $ 0Q�5�       � IsIndeterminate$
 True��5�      $��50��5�      $��50��5�      �� 5�      $��
Loading...��6   
�������$��0,10,0,0q�5�      $��Center=�5�      5�      5�      '  !MaterialDesignThemes.Wpf.Snackbar  5�   
   $��1��6"   $��Center=�5�   "   $��Bottom=�5�   "      MessageQueue 5�   "   �� *SnackbarMessageQueue+5�      5�      